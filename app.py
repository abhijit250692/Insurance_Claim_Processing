import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from datetime import date
from typing import TypedDict

load_dotenv()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# -------------------
# State
# -------------------

class GraphState(TypedDict):
    claimId: str
    date: str
    insuranceNumber: str
    insuranceStartDate: str
    insuranceEndDate: str
    sumAssured: float
    insuranceStatus: str
    billAmount: float
    claimAmount: float
    claimSummary: str
    settlementAmount: float
    claimStatus: str
    humanApprovalRequired: str
    fraudDetected: str
    eligibilityStatus: str
    humanNote: str
    humanDecision: str


# -------------------
# ClaimId generator Agent
# -------------------

def claim_id_generator_agent(state):
    claim_id = llm.invoke("Generate a 10 digit unique claim ID for the insurance claim using the insurance number: " + state["insuranceNumber"]).content
    return {"claimId": claim_id}


# -------------------
# Document Verification Agent
# -------------------

def document_verification_agent(state):
    claim_date = date.fromisoformat(state["date"])
    insurance_start_date = date.fromisoformat(state["insuranceStartDate"])
    insurance_end_date = date.fromisoformat(state["insuranceEndDate"])
    insurance_status = (
        "Active"
        if insurance_start_date <= claim_date <= insurance_end_date
        else "Inactive"
    )
    return {"insuranceStatus": insurance_status}


# -------------------
# Fraud checker Agent
# -------------------

def fraud_checker_agent(state):
    bill_amount = float(state["billAmount"])

    if bill_amount == 0:
        return {"fraudDetected": "Yes"}
    else:
        return {"fraudDetected": "No"}


# -------------------
# Eligibility checker Agent
# -------------------

def eligibility_checker_agent(state):
    insurance_status = state["insuranceStatus"]
    fraud_detected = state["fraudDetected"]

    if insurance_status == "Active" and fraud_detected == "No":
        return {"eligibilityStatus": "Eligible"}
    else:
        return {"eligibilityStatus": "Not Eligible", "claimStatus": "Rejected"}


# -------------------
# Settlement calculator Agent
# -------------------

def settlement_calculator_agent(state):
    claim_amount = float(state["claimAmount"])
    sum_assured = float(state["sumAssured"])
    bill_amount = float(state["billAmount"])

    settlement_amount = 0
    human_approval_required = "Not Required"
    claim_status = "Pending"

    if bill_amount < claim_amount:
        settlement_amount = bill_amount
    else:
        settlement_amount = claim_amount

    if settlement_amount >= sum_assured:
        settlement_amount = sum_assured

    if settlement_amount > (sum_assured * 0.8):
        human_approval_required = "Required"
    else:
        claim_status = "Approved"

    return {"settlementAmount": settlement_amount, "humanApprovalRequired": human_approval_required, "claimStatus": claim_status}


# -------------------
# Human Review Agent
# -------------------

def human_review_agent(state):
    review = interrupt(
        {
            "claimId": state["claimId"],
            "insuranceNumber": state["insuranceNumber"],
            "billAmount": state["billAmount"],
            "claimAmount": state["claimAmount"],
            "sumAssured": state["sumAssured"],
            "settlementAmount": state["settlementAmount"],
        }
    )
    result = {"claimStatus": review["decision"]}
    if review["decision"] == "Rejected":
        result["humanNote"] = review["note"]
    return result


# -------------------
# Summary Agent
# -------------------

def summary_agent(state):
    prompt = f"""
    Generate a summary of the insurance claim based on the details available in the state {state}. 
    The summary should be in a structured format.
    Amounts are in indian rupees (₹).
    If the claim is rejected, include the reason for rejection in the summary.
    No need to include any reason if claim is approved.

    Highlight claim status and settlement amount values in the summary with bold text.
    """
    summary = llm.invoke(prompt).content
    return {"claimSummary": summary}


# -------------------
# Build Graph
# -------------------

def build_claim_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("claim_id_generator", claim_id_generator_agent)
    workflow.add_node("document_verification", document_verification_agent)
    workflow.add_node("fraud_checker", fraud_checker_agent)
    workflow.add_node("eligibility_checker", eligibility_checker_agent)
    workflow.add_node("settlement_calculator", settlement_calculator_agent)
    workflow.add_node("human_review", human_review_agent)
    workflow.add_node("summary", summary_agent)

    workflow.add_edge(START, "claim_id_generator")
    workflow.add_edge("claim_id_generator", "document_verification")
    workflow.add_edge("claim_id_generator", "fraud_checker")
    workflow.add_edge("document_verification", "eligibility_checker")
    workflow.add_edge("fraud_checker", "eligibility_checker")

    workflow.add_conditional_edges(
        "eligibility_checker",
        lambda state: state["eligibilityStatus"],
        {
            "Eligible": "settlement_calculator",
            "Not Eligible": "summary"
        }
    )

    workflow.add_conditional_edges(
        "settlement_calculator",
        lambda state: state["humanApprovalRequired"],
        {
            "Required": "human_review",
            "Not Required": "summary"
        }
    )

    workflow.add_edge("human_review", "summary")
    workflow.add_edge("summary", END)
    return workflow.compile(checkpointer=MemorySaver())


app = build_claim_workflow()

st.set_page_config(page_title="Insurance Claim Processing", page_icon="🛡️")
st.title("Insurance Claim Processing")
st.write(
    "Submit claim and policy details for automated checks. "
)


@st.dialog("Additional review required")
def human_review_dialog():
    pending_claim = st.session_state.get("pending_claim")
    if pending_claim is None:
        st.info("There is no claim waiting for review.")
        return

    st.write(f"**Settlement amount:** ₹{pending_claim['settlementAmount']:,.2f}")
    st.write(f"**Bill amount:** ₹{pending_claim['billAmount']:,.2f}")
    st.write(f"**Claim amount:** ₹{pending_claim['claimAmount']:,.2f}")
    st.write(f"**Sum assured:** ₹{pending_claim['sumAssured']:,.2f}")
    with st.form("human_review_form"):
        decision = st.radio("Decision", ("Approved", "Rejected"), horizontal=True)
        note = st.text_area("Reason for rejection (required if rejecting)")
        review_submitted = st.form_submit_button("Submit decision", type="primary")

    if review_submitted:
        if decision == "Rejected" and not note.strip():
            st.error("Enter a reason before rejecting this claim.")
            return

        with st.spinner("Finalizing the claim and generating its summary..."):
            st.session_state["claim_result"] = app.invoke(
                Command(
                    resume={
                        "decision": decision,
                        "note": note.strip(),
                    }
                ),
                config={"configurable": {"thread_id": pending_claim["thread_id"]}},
            )
        del st.session_state["pending_claim"]
        st.rerun()


with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        insurance_number = st.text_input("Insurance number", value="")
        claim_date = st.date_input("Claim date", value=date.today())
        insurance_start_date = st.date_input(
            "Insurance start date", value=date(2026, 1, 1)
        )
        insurance_end_date = st.date_input(
            "Insurance end date", value=date(2026, 12, 31)
        )
    with col2:
        sum_assured = st.number_input(
            "Sum assured (₹)", min_value=0, value=500000, step=1
        )
        bill_amount = st.number_input(
            "Bill amount (₹)", min_value=0, value=100000, step=1
        )
        claim_amount = st.number_input(
            "Claim amount (₹)", min_value=0, value=100000, step=1
        )

    _, button_col, _ = st.columns([1, 2, 1])
    with button_col:
        submitted = st.form_submit_button(
            "Process claim", type="primary", use_container_width=True
        )

if submitted:
    if not insurance_number.strip():
        st.error("Enter an insurance number.")
        st.stop()
    if insurance_start_date > insurance_end_date:
        st.error("Insurance start date must be on or before the end date.")
        st.stop()

    initial_state: GraphState = {
        "claimId": "",
        "date": claim_date.isoformat(),
        "insuranceNumber": insurance_number.strip(),
        "insuranceStartDate": insurance_start_date.isoformat(),
        "insuranceEndDate": insurance_end_date.isoformat(),
        "sumAssured": sum_assured,
        "insuranceStatus": "",
        "billAmount": bill_amount,
        "claimAmount": claim_amount,
        "claimSummary": "",
        "settlementAmount": 0.0,
        "claimStatus": "Pending",
        "humanApprovalRequired": "Not Required",
        "fraudDetected": "",
        "eligibilityStatus": "",
        "humanNote": "",
        "humanDecision": "",
    }

    st.session_state.pop("claim_result", None)
    st.session_state.pop("pending_claim", None)
    thread_id = f"claim-{initial_state['insuranceNumber']}-{date.today().isoformat()}-{id(initial_state)}"
    with st.spinner("Processing the claim..."):
        result = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
    if result.get("__interrupt__"):
        st.session_state["pending_claim"] = {
            **result["__interrupt__"][0].value,
            "thread_id": thread_id,
        }
        human_review_dialog()
    else:
        st.session_state["claim_result"] = result

#if not submitted and st.session_state.get("pending_claim") is not None:
#    human_review_dialog()

result = st.session_state.get("claim_result")
if result is not None:
    st.markdown(result["claimSummary"])
