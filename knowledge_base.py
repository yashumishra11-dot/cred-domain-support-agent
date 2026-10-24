"""
Cred Domain Support Agent - Knowledge Base
Track: Banking & FinTech (Cred)
Part 1 - Task 2: Required Knowledge Base Documents (>=12 documents, 2-5 sentences each)
"""

from typing import List, Dict, Any

KNOWLEDGE_BASE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "doc_id": "KB-DOC-001",
        "topic": "loan eligibility criteria by loan type",
        "title": "Loan Eligibility Criteria Across Product Categories",
        "content": (
            "Cred evaluates loan eligibility based on applicant age, monthly net income, employment stability, and existing debt obligations. "
            "Personal Loan applicants must be salaried or self-employed individuals aged 21 to 58 years with a minimum monthly net income of INR 25,000. "
            "Home and Business Loans require a minimum operational or employment vintage of at least two years along with verifiable audited financial records. "
            "Auto and Education Loans require valid collateral or a co-borrower guarantee where the co-borrower meets the primary credit criteria."
        ),
        "metadata": {"category": "Lending Policy", "version": "1.0", "section": "Eligibility"}
    },
    {
        "doc_id": "KB-DOC-002",
        "topic": "EMI calculation rules",
        "title": "Equated Monthly Installment (EMI) Computation Guidelines",
        "content": (
            "Equated Monthly Installments (EMIs) are computed using the reducing balance method based on the sanctioned principal, agreed annual interest rate, and repayment tenor. "
            "Interest is charged on the outstanding loan balance at the close of each billing cycle rather than the original principal amount. "
            "Borrowers can elect auto-debit through NACH or e-mandate on their primary salary account with debit scheduled between the 1st and 5th of each calendar month. "
            "Prepayments or partial principal payments directly reduce the remaining loan tenor or the monthly installment amount depending on member preference."
        ),
        "metadata": {"category": "Lending Policy", "version": "1.0", "section": "Repayment"}
    },
    {
        "doc_id": "KB-DOC-003",
        "topic": "credit-card fee structure",
        "title": "Credit Card Annual Fees, Charges, and Surcharges",
        "content": (
            "Cred credit cards carry an annual membership fee ranging from zero for lifetime-free cards up to INR 4,999 for premium metal tier cards. "
            "Annual card fees are automatically waived if the cardholder achieves minimum annual retail spends of INR 150,000 within the preceding membership year. "
            "Late payment fees follow a tiered slab structure up to INR 1,200 depending on the total statement balance overdue past the grace period. "
            "Cash advances incur a transaction fee of 2.5% of the withdrawn amount subject to a minimum charge of INR 500 alongside immediate finance charges."
        ),
        "metadata": {"category": "Cards & Fees", "version": "1.0", "section": "Fee Structure"}
    },
    {
        "doc_id": "KB-DOC-004",
        "topic": "KYC document requirements",
        "title": "Know Your Customer (KYC) Verification Protocols",
        "content": (
            "Mandatory KYC compliance requires all borrowers to furnish officially valid documents including a clear copy of their Permanent Account Number (PAN) card and Aadhaar card. "
            "Proof of current residential address must be established using an Aadhaar card, valid passport, voter identity card, or utility bill not older than two months. "
            "Income verification demands the latest three months of bank account statements reflecting salary credits along with Form 16 or recent salary slips. "
            "Video KYC (V-KYC) sessions can be conducted directly inside the application during working hours for contactless, instant biometric verification."
        ),
        "metadata": {"category": "Compliance", "version": "1.0", "section": "KYC"}
    },
    {
        "doc_id": "KB-DOC-005",
        "topic": "fraud-dispute resolution process",
        "title": "Fraud Reporting and Unauthorized Transaction Dispute Resolution",
        "content": (
            "Members suspecting unauthorized transactions or identity theft must report the dispute within 72 hours through the Cred mobile app or emergency helpline. "
            "Upon notification, the affected credit line or account is immediately frozen to prevent further unauthorized financial leakage. "
            "A formal investigation docket is initiated by our risk team, and provisional credit is granted to the cardholder within seven business days for eligible claims. "
            "Final dispute adjudication is completed within a maximum window of 30 banking days in accordance with Reserve Bank of India ombudsman guidelines."
        ),
        "metadata": {"category": "Risk & Disputes", "version": "1.0", "section": "Fraud"}
    },
    {
        "doc_id": "KB-DOC-006",
        "topic": "account-closure process",
        "title": "Account and Lending Facility Closure Procedures",
        "content": (
            "A member may request formal account or loan facility closure only after all outstanding dues, interest, and pending charges are settled in full. "
            "A formal No Objection Certificate (NOC) and loan closure letter are generated and dispatched electronically within 15 business days following full repayment. "
            "Any active standing instructions, e-mandates, and collateral liens registered against the lending account are formally released upon closure confirmation. "
            "Account closure requests are permanently updated in the central credit bureau databases during the subsequent monthly reporting cycle."
        ),
        "metadata": {"category": "Customer Operations", "version": "1.0", "section": "Account Services"}
    },
    {
        "doc_id": "KB-DOC-007",
        "topic": "interest-rate slabs",
        "title": "Tiered Interest Rate Slabs by Credit Score and Loan Product",
        "content": (
            "Interest rates are risk-priced based on the applicant's bureau credit score, debt-to-income ratio, and requested loan product category. "
            "Personal Loan rates range from 10.49% per annum for credit scores above 780 up to 21.99% per annum for scores between 650 and 700. "
            "Secured Home Loan rates start at 8.35% per annum for prime tier borrowers and scale up to 10.75% based on loan-to-value (LTV) ratios. "
            "Business Loans carry floating interest rates between 13.5% and 24.0% determined by company cash flows, annual turnover, and balance sheet health."
        ),
        "metadata": {"category": "Lending Policy", "version": "1.0", "section": "Interest Rates"}
    },
    {
        "doc_id": "KB-DOC-008",
        "topic": "prepayment-penalty rules",
        "title": "Prepayment Charges and Part-Prepayment Regulations",
        "content": (
            "In strict compliance with RBI directives, floating-rate personal loans and home loans extended to individual borrowers carry zero prepayment penalty. "
            "Fixed-rate term loans and business loans carry a prepayment levy of 2% to 4% on the prepaid principal amount if foreclosed within twelve months of disbursement. "
            "Part-prepayments must equal at least two months of standard EMI amounts and cannot exceed four occurrences within any single calendar financial year. "
            "Borrowers must provide written or in-app notice at least five business days prior to executing a lump-sum loan foreclosure."
        ),
        "metadata": {"category": "Lending Policy", "version": "1.0", "section": "Prepayment"}
    },
    {
        "doc_id": "KB-DOC-009",
        "topic": "minimum-balance requirements",
        "title": "Minimum Balance Thresholds and Non-Maintenance Penalties",
        "content": (
            "Cred companion digital savings accounts require maintaining an Average Monthly Balance (AMB) of INR 10,000 for metro and urban branches. "
            "For semi-urban and rural locations, the prescribed minimum Average Monthly Balance threshold is reduced to INR 5,000. "
            "Failure to maintain the mandated minimum balance attracts a non-maintenance charge capped at 5% of the shortfall amount or INR 450, whichever is lower. "
            "Salary accounts with regular active monthly payroll credits are fully exempt from all minimum balance stipulations."
        ),
        "metadata": {"category": "Deposit Accounts", "version": "1.0", "section": "Account Balances"}
    },
    {
        "doc_id": "KB-DOC-010",
        "topic": "credit-score impact factors",
        "title": "Credit Score Calculation Drivers and Score Impact Factors",
        "content": (
            "A member's bureau credit score is influenced by repayment history, credit utilization ratio, credit history vintage, and the credit mix. "
            "Payment history accounts for the largest share at 35%, meaning a single missed EMI or credit card default can severely degrade the score. "
            "Maintaining credit utilization below 30% of aggregate sanctioned card limits is strongly recommended to protect creditworthiness. "
            "Excessive hard inquiries stemming from multiple simultaneous loan applications within a brief window negatively pull down the overall rating."
        ),
        "metadata": {"category": "Credit Advisory", "version": "1.0", "section": "Credit Bureau"}
    },
    {
        "doc_id": "KB-DOC-011",
        "topic": "joint-account rules",
        "title": "Joint Account Mandates, Survivorship Clauses, and Operations",
        "content": (
            "Cred savings and lending accounts may be opened jointly by up to three immediate family members such as spouses, parents, or adult children. "
            "Operating mandates can be configured as Either or Survivor, Former or Survivor, or Jointly depending on customer preference. "
            "Under an Either or Survivor mandate, any registered account holder possesses independent authority to operate funds and initiate transactions. "
            "In joint loan applications, both borrowers are jointly and severally liable for timely repayment of all accrued liabilities."
        ),
        "metadata": {"category": "Deposit Accounts", "version": "1.0", "section": "Joint Accounts"}
    },
    {
        "doc_id": "KB-DOC-012",
        "topic": "NRI-account eligibility",
        "title": "Non-Resident Indian (NRI) Account Eligibility and Documentation",
        "content": (
            "Non-Resident Indians (NRIs) and Persons of Indian Origin (PIOs) residing abroad for employment or business are eligible for NRE and NRO deposit facilities. "
            "Non-Resident External (NRE) accounts allow seamless tax-free repatriation of foreign earnings into India denominated in Indian Rupees. "
            "Non-Resident Ordinary (NRO) accounts are designated for managing rupee earnings accrued within India, subject to applicable Indian withholding taxes. "
            "Mandatory onboarding documentation includes a valid Indian passport, overseas residence visa or work permit, and foreign address proof attested by the Indian embassy or notary."
        ),
        "metadata": {"category": "NRI Banking", "version": "1.0", "section": "NRI Services"}
    }
]


def get_all_documents() -> List[Dict[str, Any]]:
    """Returns all 12 knowledge base documents."""
    return list(KNOWLEDGE_BASE_DOCUMENTS)


def get_document_by_id(doc_id: str) -> Dict[str, Any]:
    """Retrieves document by ID."""
    for doc in KNOWLEDGE_BASE_DOCUMENTS:
        if doc["doc_id"] == doc_id:
            return doc
    raise KeyError(f"Document {doc_id} not found in knowledge base.")


if __name__ == "__main__":
    docs = get_all_documents()
    print(f"Loaded {len(docs)} Knowledge Base documents.")
    for d in docs:
        sentences = [s for s in d["content"].split(". ") if s.strip()]
        print(f"- [{d['doc_id']}] {d['topic']}: {len(sentences)} sentences")
