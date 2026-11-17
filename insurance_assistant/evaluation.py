
"""
Evaluate answer quality against a fixed set of test questions.

Runs the agent against a known set of question/reference-answer pairs
and scores each answer using a second LLM as a judge.
Results are uploaded to LangSmith as a Dataset + Experiment,
so quality can be compared across runs.
"""

import time
from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT

from insurance_assistant import config
from insurance_assistant.logger import get_logger
from insurance_assistant.pipeline import ask, build_insurance_assistant
from insurance_assistant.vector_store import get_retriever, load_vector_store

logger = get_logger(__name__)

DATASET_NAME = config.EVAL_DATASET_NAME

TEST_CASES = [
    # Benefit limits and room rent
    {
        "question": "What is the room rent limit per day, and is there an overall cap on room, boarding and nursing expenses?",
        "answer": "Room, boarding and nursing are covered up to 1% of the Sum Insured per day, and up to 2% of the Sum Insured per day in ICU (excluding registration and service expenses). All admissible room, board and nursing claims including ICU are capped at 25% of the Sum Insured per illness/injury.",
    },
    {
        "question": "How many days of pre- and post-hospitalisation expenses are covered, and what is the limit on them?",
        "answer": "Pre-hospitalisation expenses up to 30 days before hospitalisation and post-hospitalisation expenses up to 60 days after. Each is limited to 10% of the eligible hospitalisation expenses for each admitted claim, and only for the same ailment that required hospitalisation.",
    },
    {
        "question": "What are the limits for Ayurvedic treatment and for Homeopathy or Unani treatment?",
        "answer": "Ayurvedic treatment: up to 15% of the Sum Insured per policy period, maximum INR 20,000. Homeopathy and Unani: up to 10% of the Sum Insured per policy period, maximum INR 15,000. Treatment must be taken at a hospital meeting the policy's hospital definition and registered with local government bodies.",
    },
    {
        "question": "Does the policy give a free health check-up? What are the conditions?",
        "answer": "Yes. For every four consecutive claim-free years with the insurer without a break, the insurer may arrange a free check-up at an empanelled diagnostic centre or reimburse the cost up to 1% of the Sum Insured, maximum INR 2,500.",
    },
    {
        "question": "Who is eligible for the convalescence benefit and how much is paid?",
        "answer": "Insured persons aged above 10 years and below 60 years. INR 5,000 is paid if the person is hospitalised for 10 consecutive days or more for a covered injury or illness. It is payable only once per insured during the policy period.",
    },

    # Definitions and hospital requirements
    {
        "question": "What counts as hospitalisation under this policy?",
        "answer": "Admission into a hospital for a continuous period of not less than 24 hours.",
    },
    {
        "question": "What is the minimum number of inpatient beds a hospital must have to qualify under this policy?",
        "answer": "At least 10 inpatient beds in towns with a population of less than 10,00,000 and at least 15 inpatient beds in all other places (along with round-the-clock nursing staff and medical practitioner, and its own operation theatre), unless it is registered with local authorities and supervised by a registered medical practitioner.",
    },
    {
        "question": "Which Insurance Ombudsman office should I contact in Ahmedabad, and what is its jurisdiction?",
        "answer": "Office of the Insurance Ombudsman, Jeevan Prakash Building, 6th floor, Tilak Marg, Relief Road, Ahmedabad - 380 001. Tel: 079-25501201/02/05/06, email bimalokpal.ahmedabad@ecoi.co.in. Jurisdiction: Gujarat, Dadra & Nagar Haveli, Daman and Diu.",
    },

    # Policy exclusions
    {
        "question": "Are pregnancy and caesarean delivery expenses covered?",
        "answer": "No. Treatment arising from pregnancy, childbirth, miscarriage, abortion or their complications, including caesarean section, is excluded. The only exception is an abdominal operation for ectopic pregnancy, proved by an ultrasonography report and a gynaecologist's certificate that it is life threatening.",
    },
    {
        "question": "Is treatment for mental illness or psychiatric conditions covered?",
        "answer": "No. Treatment for any mental illness or psychiatric or psychological ailment is excluded, as is treatment at an institution primarily for psychiatric or mental disorders.",
    },
    {
        "question": "Will the policy pay for injuries from scuba diving or paragliding?",
        "answer": "No. Treatment arising from hazardous activities such as skiing, scuba diving, motor racing, parachuting, hang gliding and rock or mountain climbing is excluded unless specifically agreed by the insurer.",
    },
    {
        "question": "Can I claim for surgery done in a hospital in Singapore?",
        "answer": "No. Treatment outside India is excluded; all treatment must be taken in India and claims are payable in Indian currency.",
    },
    {
        "question": "Is naturopathy treatment covered, and how is it different from Ayurveda under this policy?",
        "answer": "Naturopathy is excluded (along with acupuncture, acupressure, osteopathy, chiropractic, reflexology, aromatherapy, and treatment in health hydros, spas and nature care clinics). Ayurveda, Homeopathy and Unani are covered but with sub-limits.",
    },

    # Waiting periods
    {
        "question": "What is the waiting period for pre-existing diseases?",
        "answer": "48 months (4 years). Pre-existing conditions are excluded until 48 consecutive months have elapsed since inception of the first policy with the insurer; the exclusion ceases after 4 continuous years without a break.",
    },
    {
        "question": "I have been insured continuously for 3 years with another insurer and am porting to SBI General. What happens to the pre-existing disease waiting period?",
        "answer": "The pre-existing condition stands waived after a waiting period of 1 year from commencement of the SBI General policy, and only up to the sum insured of the expiring policy.",
    },

    # Coverage scenarios
    {
        "question": "I took my first policy 18 months ago. I was first diagnosed with diabetes yesterday and need hospitalisation for its complications. Is it covered?",
        "answer": "No. Diabetes and related complications are excluded during the first two years of cover from the commencement of the policy, so month 18 falls within the exclusion.",
    },
    {
        "question": "I need hernia surgery 8 months after buying my first policy. Will it be paid?",
        "answer": "No. All types of hernia and hydrocele are excluded during the first year of cover. It would become payable only after the first year, subject to the other policy terms.",
    },
    {
        "question": "In month 20 of my first policy I need a knee replacement because of age-related osteoarthritis. Is it covered? What if the knee was damaged in an accident?",
        "answer": "Not covered if due to degenerative/age-related osteoarthritis or osteoporosis, because joint replacement is excluded for the first three years. If the replacement is necessitated by accidental bodily injury, the exclusion does not apply and it can be covered subject to other terms.",
    },
    {
        "question": "I bought my first policy and 20 days later was hospitalised with typhoid. Separately, a friend on a new policy was hospitalised on day 10 after a road accident. Which of these claims is payable?",
        "answer": "The typhoid claim is not payable because illness within 30 days of the first policy is excluded. The accident claim is payable because the 30-day waiting period does not apply to accidental bodily injury.",
    },

    # Claims submission and procedures
    {
        "question": "My father was admitted in an emergency and we informed the insurer only on day 5 because the family was away. Will the claim be rejected?",
        "answer": "The insurer must be informed immediately and within 48 hours of hospitalisation. However, the insurer may at its sole discretion relax this up to a maximum of 7 days if a justifiable reason or evidence for the delay is produced, so it is not automatically rejected but it is not guaranteed either.",
    },
    {
        "question": "What are the deadlines for submitting a reimbursement claim and a post-hospitalisation claim?",
        "answer": "Claim documents must be filed within 15 days of discharge. Post-hospitalisation claims must be lodged within 15 days of completing post-hospitalisation treatment, subject to a maximum of 75 days from the date of discharge. Delayed submission without a justified reason may lead to the claim not being considered.",
    },
    {
        "question": "Which documents do I need to submit for a claim?",
        "answer": "Duly filled claim form, valid photo identity card, original discharge card/certificate/death summary, copies of prescriptions for diagnostic tests, treatment advice and medical references, original set of investigation reports, itemised original hospital bill and receipts, and original pharmacy bills with prescriptions.",
    },
    {
        "question": "My policy expired 20 days ago and I did not renew. If I renew now, do I keep my waiting period credits?",
        "answer": "No. Renewal after 15 days from the expiry date is treated as a fresh policy and all terms, conditions and waiting periods apply again. Continuity benefits are kept only if the policy is renewed with the insurer's acceptance within 15 days of expiry, and even then there is no cover for the break period.",
    },

    # Sub-limit and payout calculations
    {
        "question": "My Sum Insured is Rs. 5,00,000. What is the maximum room rent per day for a normal room and for ICU, the total cap on room, boarding and nursing per illness, and the cap on doctors' fees per illness?",
        "answer": "Normal room: Rs. 5,00,000 Sum Insured gives Rs. 5,000 per day (1%). ICU: Rs. 10,000 per day (2%). Room, board and nursing cap per illness: Rs. 1,25,000 (25%). Medical practitioner fees cap per illness: Rs. 2,00,000 (40%).",
    },
    {
        "question": "With a Sum Insured of Rs. 5,00,000, what is the maximum payable for cataract per eye, for ambulance, and for Ayurvedic treatment?",
        "answer": "Cataract: Rs. 25,000 per eye (15% would be Rs. 75,000 but it is capped at Rs. 25,000). Ambulance: Rs. 1,500 (1% would be Rs. 5,000 but it is capped at Rs. 1,500). Ayurvedic: Rs. 20,000 (15% would be Rs. 75,000 but it is capped at Rs. 20,000).",
    },
    {
        "question": "If my Sum Insured is only Rs. 1,00,000, how much will the policy pay for ambulance and for cataract surgery per eye?",
        "answer": "Ambulance: Rs. 1,000 (1% of Sum Insured, which is below the Rs. 1,500 cap). Cataract: Rs. 15,000 per eye (15% of Sum Insured, which is below the Rs. 25,000 cap), and subject to the two-year cataract exclusion.",
    },
    {
        "question": "My Sum Insured is Rs. 5,00,000 with no previous claims and I have an accident. What Sum Insured is available for that hospitalisation?",
        "answer": "Rs. 6,00,000. The accidental hospitalisation benefit raises the limit to 125% of the available Sum Insured (Rs. 6,25,000), but the increase over the base Sum Insured is capped at Rs. 1,00,000, so the total is Rs. 6,00,000. The benefit is payable once per insured person during the policy period.",
    },
    {
        "question": "I was treated in a non-network hospital and my admissible claim is Rs. 2,00,000. Assuming no excess, how much do I pay and how much does the insurer pay?",
        "answer": "A 10% co-pay applies in non-network hospitals, so the insured bears Rs. 20,000 and the insurer pays Rs. 1,80,000.",
    },
    {
        "question": "I live in Ahmedabad and have Plan B. I was treated in Mumbai and my admissible claim is Rs. 1,00,000. How much will be paid? And if I had been treated in Chennai?",
        "answer": "Ahmedabad is a Location B city and Mumbai is Location A. Plan B pays 80% for Location A treatment, so Rs. 80,000 is payable. In Chennai (Location B) Plan B pays 100%, so Rs. 1,00,000.",
    },

    # Day care and multi-condition rules
    {
        "question": "I was admitted for only 18 hours for IV antibiotics for an infection and then discharged. Can I claim?",
        "answer": "Generally no. Hospitalisation requires a continuous stay of at least 24 hours. A shorter stay is payable only if it is a day care procedure listed in Annexure A that was pre-authorised by the insurer or administrator. IV antibiotics for an infection is not in that list.",
    },
    {
        "question": "Tonsillectomy is in the day care list. Will it be paid if I get it done in the 6th month of my first policy?",
        "answer": "No. Although tonsillectomy appears in the Annexure A day care list, it is excluded during the first year of cover from policy commencement, so it would not be payable in month 6.",
    },

    # Clarifications and unanswerable checks
    {
        "question": "Since maternity is covered after a 2-year waiting period, how much can I claim for delivery in the third year?",
        "answer": "The premise is wrong. This policy has no maternity benefit; pregnancy, childbirth and caesarean section are excluded outright, not subject to a waiting period. Only an abdominal operation for life-threatening ectopic pregnancy is an exception.",
    },
    {
        "question": "What is the annual premium for a 35 year old with a Sum Insured of Rs. 5 lakh?",
        "answer": "Premium rates are not in the policy document (the Sum Insured and premium are in the Schedule). The system should say it cannot find this and not invent a number.",
    },
    {
        "question": "What is SBI General's claim settlement ratio?",
        "answer": "Not in the policy document. The system should say it cannot find this and not quote any percentage.",
    },
    {
        "question": "Is this policy better than Star Health Comprehensive?",
        "answer": "The document cannot support a comparison with another insurer's product. The system should decline to compare and may only describe this policy's own terms.",
    },
]


def _get_judge_llm() -> ChatOpenAI:
    """Return a judge model configured from environment variables with retry handling."""
    raw_model = config.JUDGE_MODEL_NAME or "gemini-3.1-flash-lite"
    model_id = raw_model.strip().lower().replace(" ", "-")
    logger.info("Initializing Judge LLM (model: %s)", model_id)
    return ChatOpenAI(
        api_key=config.JUDGE_API_KEY,
        base_url=config.JUDGE_BASE_URL,
        model=model_id,
        temperature=0,
        max_retries=6,
    )


def _ensure_dataset(client: Client):
    """Create the LangSmith dataset if it doesn't exist yet, and upload the test cases."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)", DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="PolicyScan insurance policy question/answer evaluation dataset",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": case["question"]},
                "outputs": {"answer": case["answer"]},
            }
            for case in TEST_CASES
        ],
    )
    return dataset


def run_evaluation(experiment_prefix: str = "policyscan-eval"):
    """Upload the dataset (if needed) and run the correctness + groundedness evaluation with rate limit protection."""
    client = Client()
    dataset = _ensure_dataset(client)

    agent = build_insurance_assistant()
    retriever = get_retriever(load_vector_store())

    def target(inputs: dict) -> dict:
        """
        Run one test question through the real agent with retry backoff,
        capture retrieved context, and apply inter-case pacing to protect 8K TPM.
        """
        question = inputs["question"]
        logger.info("Evaluating test question: %s", question)

        # Retry with exponential backoff on Groq 8K TPM rate limits
        answer = None
        for attempt in range(1, 6):
            try:
                answer = ask(agent, question)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "limit" in err_str or "tpm" in err_str:
                    wait_sec = 25 * attempt
                    logger.warning(
                        "RAG pipeline hit rate limit on '%s' (attempt %d/5): %s. Backing off for %ds...",
                        question,
                        attempt,
                        e,
                        wait_sec,
                    )
                    time.sleep(wait_sec)
                else:
                    logger.error("Error during RAG execution: %s", e)
                    raise
        if answer is None:
            answer = "Error: RAG pipeline rate limit retry exceeded."

        chunks = retriever.invoke(question)
        context = "\n\n".join(chunk.page_content for chunk in chunks)

        # Inter-case pacing sleep to strictly respect Groq 8K TPM & Gemini 15 RPM limits
        pacing_delay = getattr(config, "EVAL_PACING_DELAY", 20.0)
        if pacing_delay > 0:
            logger.info("Pacing delay: sleeping %.1fs to protect 8K TPM and 15 RPM quotas...", pacing_delay)
            time.sleep(pacing_delay)

        return {"answer": answer, "context": context}

    judge_llm = _get_judge_llm()

    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=judge_llm,
    )

    def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict, **kwargs) -> dict:
        """Evaluate correctness with rate-limit pacing and retry protection."""
        time.sleep(2.0)
        for attempt in range(1, 6):
            try:
                return correctness_judge(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource" in err_str:
                    wait_time = 15 * attempt
                    logger.warning("Correctness evaluator rate limited (attempt %d/5): %s. Waiting %ds...", attempt, e, wait_time)
                    time.sleep(wait_time)
                else:
                    raise
        return {"key": "correctness", "score": False, "comment": "Evaluation aborted due to repeated rate limits"}

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=judge_llm,
    )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        """Check the answer is supported by the retrieved context with rate-limit pacing and retry protection."""
        delay = getattr(config, "EVAL_JUDGE_DELAY", 5.0)
        if delay > 0:
            time.sleep(delay)
        for attempt in range(1, 6):
            try:
                return groundedness_judge(outputs={"answer": outputs["answer"]}, context=outputs["context"], **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource" in err_str:
                    wait_time = 15 * attempt
                    logger.warning("Groundedness evaluator rate limited (attempt %d/5): %s. Waiting %ds...", attempt, e, wait_time)
                    time.sleep(wait_time)
                else:
                    raise
        return {"key": "groundedness", "score": False, "comment": "Evaluation aborted due to repeated rate limits"}

    logger.info("Running evaluation against dataset '%s' with max_concurrency=1", DATASET_NAME)
    return client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix=experiment_prefix,
        description="PolicyScan insurance policy assistant correctness + groundedness evaluation",
        max_concurrency=1,
    )

