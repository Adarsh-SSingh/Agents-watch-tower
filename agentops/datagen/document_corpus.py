"""Realistic synthetic document corpus + question set for the Document Agent.

Every document is generated from known facts, so every question has a known
answer. That gives a ground-truth label ("did the agent answer correctly?")
without any manual labelling.

14 document types that look like real-life files: rent agreement, offer
letter, insurance policy, EMI schedule (CSV table), sales report (CSV table),
syllabus, job description, resume, product manual (long), API documentation,
annual report (with tables), meeting transcript (long), support FAQ,
government circular.

All names, companies and numbers are invented. No real person is described.

Question types:
  Fact          one fact is stated in the document
  Multi         two facts in one question
  Table Lookup  one cell of a table
  Calculation   a small calculation on facts of the document
  Aggregation   needs MANY rows (sum / maximum) - hard for top-k retrieval
  Yes No        a clause question with a yes/no answer
  Unanswerable  the document does NOT contain the answer
"""

import random

FIRST = ["Aarav", "Diya", "Rohan", "Meera", "Kabir", "Ananya", "Vikram", "Sneha",
         "Arjun", "Isha", "Neel", "Pooja", "Rahul", "Tanvi", "Yash", "Kavya",
         "Dev", "Riya", "Karan", "Nisha", "Aditya", "Simran", "Varun", "Tara"]
LAST = ["Sharma", "Verma", "Iyer", "Nair", "Singh", "Reddy", "Gupta", "Mehta",
        "Joshi", "Patel", "Rao", "Khanna", "Bose", "Menon", "Das", "Kulkarni",
        "Chopra", "Pillai", "Saxena", "Bhatt", "Malhotra", "Sinha", "Jain", "Kapoor"]
COMPANIES = ["Brightwave Technologies", "Nimbus Labs", "Orchid Systems", "Kestrel Works",
             "Lumen Analytics", "Cobalt Retail", "Ardent Logistics", "Zephyr Foods",
             "Pinnacle Health", "Saffron Digital", "Granite Infra", "Mosaic Learning",
             "Harbor Finserv", "Tundra Energy", "Quartz Mobility", "Vertex Pharma"]
CITIES = ["Bengaluru", "Pune", "Hyderabad", "Chennai", "Mumbai", "Jaipur", "Kolkata",
          "Ahmedabad", "Lucknow", "Indore", "Kochi", "Nagpur"]
LOCALITIES = ["Indiranagar", "Kothrud", "Madhapur", "Adyar", "Powai", "Malviya Nagar",
              "Salt Lake", "Satellite", "Gomti Nagar", "Vijay Nagar", "Edappally", "Dharampeth"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
SKILLS = ["Python", "SQL", "Docker", "Kubernetes", "Java", "React", "Tableau", "Spark",
          "TensorFlow", "PyTorch", "Git", "AWS", "Linux", "FastAPI", "Airflow", "Excel"]
DEGREES = ["B.Tech in Computer Science", "B.Tech in Electronics", "BCA", "B.Sc in Statistics",
           "MBA in Finance", "M.Tech in Data Science", "B.Com", "MCA"]
COLLEGES = ["Lakeview Institute of Technology", "Northfield University", "Sunrise College of Engineering",
            "Riverdale Institute", "Hilltop University", "Crescent College of Science"]


def person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def date(rng, year=None):
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {year or rng.choice([2025, 2026])}"


def money(value):
    return f"{value:,}"


def q(qtype, question, expected, difficulty, absent=None):
    """One question. `absent`: words that must NOT occur in the document."""
    return {"question_type": qtype, "question": question, "expected": expected,
            "difficulty": difficulty, "absent": absent or []}



GENERIC = [
    "This section is provided for general guidance and may be updated from time to time.",
    "Users are advised to keep a copy of this document for future reference.",
    "In case of any difference between this summary and the official records, the official records will prevail.",
    "All dates and times mentioned follow Indian Standard Time unless stated otherwise.",
    "Any change to these terms will be communicated in writing at least fifteen days in advance.",
    "Both parties agree to act in good faith and to resolve minor differences through discussion.",
    "Records related to this matter are maintained for a period of seven years.",
    "Please keep your registered contact details up to date to receive important notices.",
    "Reasonable care should be taken at all times while following the instructions given here.",
    "Complaints, if any, should be raised in writing and will be acknowledged within a few working days.",
    "The information provided here is correct to the best of our knowledge at the time of preparation.",
    "Where a step is unclear, please refer to the relevant earlier section before proceeding.",
]


def extra_sections(rng, topics, count, intro, sentences=3):
    """Long, realistic-looking sections that carry no question facts.
    They make documents longer (more chunks) and add confusing near-matches."""
    blocks = []
    for number in range(1, count + 1):
        topic = rng.choice(topics)
        body = " ".join(rng.sample(GENERIC, sentences))
        detail = f"{intro} {topic} item {number}: reference limit {rng.randint(5, 95)} units, review cycle {rng.randint(2, 12)} weeks."
        blocks.append(("h", f"Section {number}: {topic.title()}"))
        blocks.append(("p", f"{detail} {body}"))
    return blocks


# Blocks: ("h", text) heading, ("p", text) paragraph,
#         ("table", header, rows), ("csv", header, rows)

# ---------------------------------------------------------------- builders

def rent_agreement(rng):
    landlord, tenant = person(rng), person(rng)
    city, locality = rng.choice(CITIES), rng.choice(LOCALITIES)
    flat = f"Flat {rng.randint(101, 909)}, {locality}, {city}"
    rent = rng.randrange(12, 60) * 1000
    months_dep = rng.choice([2, 3, 6])
    deposit = rent * months_dep
    lockin = rng.choice([6, 11, 12])
    notice = rng.choice([1, 2, 3])
    escal = rng.choice([5, 8, 10])
    start = date(rng)
    maint = rng.randrange(1, 6) * 500

    blocks = [
        ("h", "Residential Rent Agreement"),
        ("p", f"This agreement is made on {start} between {landlord} (the Landlord) and {tenant} "
              f"(the Tenant) for the residential property at {flat}."),
        ("h", "1. Rent and Deposit"),
        ("p", f"The monthly rent is INR {money(rent)}, payable on or before the 5th day of each month. "
              f"The Tenant has paid a refundable security deposit of INR {money(deposit)}, equal to "
              f"{months_dep} months of rent. The deposit carries no interest."),
        ("p", f"Monthly maintenance charges of INR {money(maint)} are payable separately to the housing society."),
        ("h", "2. Term and Lock-in"),
        ("p", f"The tenancy starts on {start}. There is a lock-in period of {lockin} months during which "
              f"neither party may terminate the agreement. After the lock-in, either party may end the "
              f"agreement by giving {notice} months written notice."),
        ("p", f"The rent will increase by {escal} percent at the end of every twelve months of tenancy."),
        ("h", "3. Use of the Premises"),
        ("p", "The premises shall be used only for residential purposes. The Tenant shall not carry on any "
              "business from the premises and shall not store any dangerous or illegal material."),
        ("p", "Subletting or sharing the premises with persons other than the Tenant's immediate family "
              "is strictly prohibited without written permission of the Landlord."),
        ("h", "4. Repairs and Maintenance"),
        ("p", "Minor repairs such as replacing bulbs, taps and fuses are the responsibility of the Tenant. "
              "Structural repairs and repairs to plumbing or wiring caused by normal wear are the "
              "responsibility of the Landlord."),
        ("h", "5. Termination and Return of Deposit"),
        ("p", "On vacating the premises the Tenant shall return the keys and the Landlord shall refund the "
              "security deposit after deducting any unpaid rent, electricity dues or cost of damage "
              "beyond normal wear and tear."),
        ("p", "Both parties have read and understood this agreement and have signed it in the presence "
              "of two witnesses."),
    ] + extra_sections(rng, ["electricity and water", "society rules", "visitors", "pets policy", "garbage disposal",
                             "fire safety", "inspection of premises", "dispute resolution"], 8, "Clause on")
    tag = f"the rent agreement of {tenant}"
    qa = [
        q("Fact", f"What is the monthly rent in {tag}?", [str(rent)], "easy"),
        q("Fact", f"Who is the landlord in {tag}?", [landlord], "easy"),
        q("Fact", f"What is the lock-in period in {tag}?", [str(lockin)], "medium"),
        q("Fact", f"How many months notice is required to end {tag} after the lock-in?", [str(notice)], "medium"),
        q("Fact", f"By what percentage does the rent increase every year in {tag}?", [str(escal)], "medium"),
        q("Fact", f"What is the security deposit amount in {tag}?", [str(deposit)], "medium"),
        q("Multi", f"What are the monthly rent and the maintenance charge in {tag}?", [str(rent), str(maint)], "medium"),
        q("Yes No", f"Is subletting allowed in {tag}?", ["no"], "medium"),
        q("Calculation", f"What will the monthly rent be after the first yearly increase in {tag}?",
          [str(rent * (100 + escal) // 100)], "hard"),
        q("Calculation", f"What is the total of the security deposit and the first month's rent in {tag}?",
          [str(deposit + rent)], "hard"),
        q("Unanswerable", f"Is a parking space included in {tag}?", [], "hard", ["parking"]),
        q("Unanswerable", f"What is the landlord's phone number in {tag}?", [], "hard", ["phone", "mobile"]),
    ]
    return f"Rent Agreement - {tenant}", blocks, qa


def offer_letter(rng):
    company, candidate = rng.choice(COMPANIES), person(rng)
    role = rng.choice(["Data Analyst", "Software Engineer", "Business Analyst", "ML Engineer",
                       "QA Engineer", "Product Associate"])
    ctc_lakh = rng.randrange(4, 24)
    ctc = ctc_lakh * 100000
    joining = date(rng)
    city = rng.choice(CITIES)
    probation = rng.choice([3, 6])
    notice = rng.choice([30, 60, 90])
    variable = rng.choice([8, 10, 12, 15])
    bonus = rng.choice([0, 0, 25000, 50000])
    insurance = rng.choice([300000, 500000])

    blocks = [
        ("h", f"{company} - Letter of Offer"),
        ("p", f"Dear {candidate}, we are pleased to offer you the position of {role} at {company}. "
              f"Your place of work will be {city}, and your expected date of joining is {joining}."),
        ("h", "Compensation"),
        ("p", f"Your annual cost to company (CTC) will be INR {money(ctc)}, of which {variable} percent "
              f"is variable pay linked to company and individual performance."),
    ]
    if bonus:
        blocks.append(("p", f"You will also receive a one time joining bonus of INR {money(bonus)}, payable "
                            f"with your first salary. If you leave within twelve months, the bonus must be repaid."))
    blocks += [
        ("h", "Probation and Notice"),
        ("p", f"You will be on probation for {probation} months from the date of joining. After confirmation, "
              f"either party may end the employment by giving {notice} days notice."),
        ("h", "Benefits"),
        ("p", f"You and your dependants will be covered by group health insurance of INR {money(insurance)}. "
              "You are entitled to paid leave as per company policy."),
        ("h", "Conditions"),
        ("p", "This offer is subject to satisfactory background verification and submission of your "
              "educational and experience certificates on the date of joining."),
        ("p", "Please sign and return a copy of this letter as acceptance of the offer within seven days."),
    ]
    tag = f"the offer letter of {candidate}"
    monthly = ctc // 12
    qa = [
        q("Fact", f"What is the annual CTC in {tag}?", [str(ctc)], "easy"),
        q("Fact", f"What position is offered in {tag}?", [role], "easy"),
        q("Fact", f"What is the date of joining in {tag}?", [joining], "easy"),
        q("Fact", f"What is the notice period in {tag}?", [str(notice)], "medium"),
        q("Fact", f"How long is the probation period in {tag}?", [str(probation)], "medium"),
        q("Fact", f"What percentage of the CTC is variable pay in {tag}?", [str(variable)], "medium"),
        q("Multi", f"What are the place of work and the health insurance cover in {tag}?", [city, str(insurance)], "medium"),
        q("Yes No", f"Is there a joining bonus in {tag}?", ["yes" if bonus else "no"], "medium"),
        q("Calculation", f"What is the monthly CTC (annual CTC divided by 12) in {tag}?", [str(monthly)], "hard"),
        q("Unanswerable", f"Does {tag} include stock options?", [], "hard", ["stock option", "esop", "equity"]),
        q("Unanswerable", f"What is the name of the reporting manager in {tag}?", [], "hard", ["reporting manager", "will report"]),
    ]
    return f"Offer Letter - {candidate}", blocks, qa


def insurance_policy(rng):
    insurer = rng.choice(["Suraksha Life", "Bharat Shield", "Care Plus", "Medisure", "Aarogya Mutual"])
    plan = f"{insurer} {rng.choice(['Silver', 'Gold', 'Platinum', 'Family Care'])} Health Plan"
    cover = rng.choice([300000, 500000, 1000000, 1500000])
    premium = rng.randrange(6, 40) * 1000
    wait = rng.choice([24, 36, 48])
    copay = rng.choice([0, 10, 20])
    room = rng.choice([3000, 4000, 5000])
    intimate = rng.choice([24, 48])
    term = rng.choice([1, 2, 3])

    blocks = [
        ("h", f"{plan} - Policy Summary"),
        ("p", f"This summary describes the main benefits of the {plan} offered by {insurer}. "
              "Please read the full policy wording before buying."),
        ("h", "Coverage"),
        ("p", f"The sum insured is INR {money(cover)} per policy year. The policy term is {term} year(s). "
              f"The annual premium is INR {money(premium)} including taxes."),
        ("p", f"Room rent is covered up to INR {money(room)} per day. A co-payment of {copay} percent applies "
              "to every claim." if copay else
              f"Room rent is covered up to INR {money(room)} per day. No co-payment applies to claims."),
        ("h", "Waiting Periods"),
        ("p", f"Pre-existing conditions are covered only after a waiting period of {wait} months from the "
              "policy start date. Claims for accidents are covered from day one."),
        ("h", "Exclusions"),
        ("p", "The following are not covered: cosmetic or plastic surgery without medical need, dental "
              "treatment unless caused by an accident, injuries from adventure sports, and treatment "
              "taken outside India."),
        ("h", "Claims"),
        ("p", f"The insurer must be informed of a planned hospitalisation at least {intimate} hours before "
              "admission. For emergencies, intimation is required within 24 hours after admission."),
        ("p", "Cashless treatment is available at network hospitals. At other hospitals, the insured can "
              "claim reimbursement by submitting original bills within 15 days of discharge."),
    ] + extra_sections(rng, ["renewal", "portability", "grievance redressal", "free look period", "ayush treatment",
                             "day care procedures", "ambulance charges", "documents for claim"], 10, "Note on")
    tag = f"the {plan}"
    qa = [
        q("Fact", f"What is the sum insured of {tag}?", [str(cover)], "easy"),
        q("Fact", f"What is the annual premium of {tag}?", [str(premium)], "easy"),
        q("Fact", f"What is the waiting period for pre-existing conditions in {tag}?", [str(wait)], "medium"),
        q("Fact", f"What is the maximum room rent per day covered by {tag}?", [str(room)], "medium"),
        q("Fact", f"How many hours before admission must the insurer be informed of a planned hospitalisation under {tag}?", [str(intimate)], "medium"),
        q("Fact", f"What is the co-payment percentage in {tag}?", [str(copay)], "medium"),
        q("Multi", f"What are the sum insured and the policy term of {tag}?", [str(cover), str(term)], "medium"),
        q("Yes No", f"Is cosmetic surgery without medical need covered by {tag}?", ["no"], "medium"),
        q("Calculation", f"How much premium is paid in total over 3 years for {tag}, if the premium stays the same?", [str(premium * 3)], "hard"),
        q("Unanswerable", f"Is maternity treatment covered by {tag}?", [], "hard", ["maternity", "pregnan"]),
        q("Unanswerable", f"What is the agent commission for {tag}?", [], "hard", ["commission"]),
    ]
    return f"Policy Summary - {plan}", blocks, qa


def emi_schedule(rng):
    loan_id = f"LN-{rng.randint(1000, 9999)}"
    amount = rng.randrange(2, 20) * 100000
    rate = rng.choice([8, 9, 10, 11, 12])
    tenure = rng.choice([24, 36, 48, 60])
    r = rate / 12 / 100
    emi = round(amount * r * (1 + r) ** tenure / ((1 + r) ** tenure - 1))
    balance = amount
    rows = []
    for month in range(1, 37):
        interest = round(balance * r)
        principal = emi - interest
        balance -= principal
        rows.append([loan_id, month, emi, principal, interest, balance])
    header = ["Loan ID", "Month", "EMI", "Principal Paid", "Interest Paid", "Closing Balance"]
    tag = f"loan {loan_id}"
    total_interest = sum(row[4] for row in rows[:12])
    m_a, m_b = rng.sample(range(1, 37), 2)
    qa = [
        q("Table Lookup", f"What is the EMI of {tag}?", [str(emi)], "easy"),
        q("Table Lookup", f"What is the closing balance after month {m_a} of {tag}?", [str(rows[m_a - 1][5])], "medium"),
        q("Table Lookup", f"How much interest is paid in month {m_b} of {tag}?", [str(rows[m_b - 1][4])], "medium"),
        q("Table Lookup", f"How much principal is paid in month {m_a} of {tag}?", [str(rows[m_a - 1][3])], "medium"),
        q("Aggregation", f"What is the total interest paid over the first 12 months in {tag}?", [str(total_interest)], "hard"),
        q("Aggregation", f"Among months 1 to 12 of {tag}, in which month is the interest paid the lowest?", ["12"], "hard"),
        q("Unanswerable", f"What is the processing fee of {tag}?", [], "hard", ["processing fee"]),
    ]
    return f"EMI Schedule - {loan_id}", [("csv", header, rows)], qa


def sales_report(rng):
    report = f"SR-{rng.randint(100, 999)}"
    regions = ["North", "South", "East", "West"]
    months = list(MONTHS)
    rows, rev = [], {}
    for region in regions:
        for month in months:
            units = rng.randint(80, 600)
            price = rng.choice([450, 500, 550, 600])
            revenue = units * price
            rev[(region, month)] = revenue
            rows.append([report, region, month, units, revenue])
    header = ["Report ID", "Region", "Month", "Units Sold", "Revenue"]
    region_a, region_b = rng.sample(regions, 2)
    month_a = rng.choice(months)
    total_a = sum(v for (rg, _), v in rev.items() if rg == region_a)
    best_region = max(regions, key=lambda rg: rev[(rg, month_a)])
    tag = f"sales report {report}"
    qa = [
        q("Table Lookup", f"What is the revenue of the {region_a} region in {month_a} in {tag}?", [str(rev[(region_a, month_a)])], "medium"),
        q("Table Lookup", f"How many units were sold in the {region_b} region in {months[0]} in {tag}?",
          [str(next(r[3] for r in rows if r[1] == region_b and r[2] == months[0]))], "medium"),
        q("Aggregation", f"What is the total revenue of the {region_a} region over all months in {tag}?", [str(total_a)], "hard"),
        q("Aggregation", f"Which region had the highest revenue in {month_a} in {tag}?", [best_region], "hard"),
        q("Unanswerable", f"What is the profit margin in {tag}?", [], "hard", ["profit", "margin"]),
    ]
    return f"Sales Report - {report}", [("csv", header, rows)], qa


def syllabus(rng):
    code = f"{rng.choice(['CS', 'DS', 'EC', 'MG'])}{rng.randint(201, 499)}"
    subject = rng.choice(["Machine Learning", "Database Systems", "Operating Systems", "Computer Networks",
                          "Data Structures", "Digital Marketing", "Statistics for Data Science"])
    credits = rng.choice([3, 4])
    internal, external = rng.choice([(30, 70), (40, 60), (50, 50)])
    exam = date(rng, 2026)
    book = f"{rng.choice(['Fundamentals of', 'Principles of', 'Introduction to'])} {subject} by {person(rng)}"
    units = [
        (f"Unit {i}", ", ".join(rng.sample(
            ["introduction", "basic concepts", "case studies", "algorithms", "evaluation methods",
             "tools and libraries", "applications", "ethics", "optimisation", "design patterns",
             "performance analysis", "security basics"], 3)))
        for i in range(1, 6)
    ]
    unit_no = rng.randint(1, 5)
    unit_topics = units[unit_no - 1][1].split(", ")[0]
    blocks = [
        ("h", f"{code} {subject} - Course Outline"),
        ("p", f"Course code: {code}. Credits: {credits}. This course introduces the main ideas of {subject}."),
        ("h", "Evaluation"),
        ("p", f"The internal assessment carries {internal} marks and the end semester examination carries "
              f"{external} marks. The end semester examination will be held on {exam}."),
        ("h", "Units"),
    ] + [("p", f"{u}: {topics}.") for u, topics in units] + [
        ("h", "Textbook"),
        ("p", f"Prescribed textbook: {book}."),
        ("h", "Attendance"),
        ("p", "A minimum attendance of 75 percent is required to appear in the end semester examination."),
    ] + extra_sections(rng, ["laboratory work", "assignments", "mini project", "guest lectures", "reference reading",
                             "course outcomes"], 5, "Details of")
    tag = f"the {code} course"
    qa = [
        q("Fact", f"How many credits does {tag} have?", [str(credits)], "easy"),
        q("Fact", f"When is the end semester examination of {tag}?", [exam], "easy"),
        q("Fact", f"How many marks does the internal assessment carry in {tag}?", [str(internal)], "medium"),
        q("Fact", f"What is the prescribed textbook of {tag}?", [book.split(" by ")[0]], "medium"),
        q("Fact", f"What minimum attendance is required in {tag}?", ["75"], "medium"),
        q("Fact", f"Which topics are listed in Unit {unit_no} of {tag}?", [unit_topics], "medium"),
        q("Calculation", f"What are the total marks (internal plus external) of {tag}?", [str(internal + external)], "hard"),
        q("Unanswerable", f"What is the email address of the instructor of {tag}?", [], "hard", ["email", "e-mail"]),
        q("Unanswerable", f"What is the fee for {tag}?", [], "hard", ["fee"]),
    ]
    return f"Course Outline - {code}", blocks, qa


def job_description(rng):
    company = rng.choice(COMPANIES)
    role = rng.choice(["Data Scientist", "Backend Developer", "Data Engineer", "DevOps Engineer",
                       "Frontend Developer", "Machine Learning Engineer"])
    low = rng.randint(1, 4)
    high = low + rng.randint(2, 4)
    city = rng.choice(CITIES)
    days = rng.randint(2, 4)
    must = rng.sample(SKILLS, 4)
    good = rng.sample([s for s in SKILLS if s not in must], 3)
    remote = rng.choice([True, False])
    openings = rng.randint(2, 12)
    blocks = [
        ("h", f"{role} - {company}"),
        ("p", f"{company} is hiring a {role} for its {city} office. We have {openings} open positions."),
        ("h", "Experience"),
        ("p", f"We are looking for candidates with {low} to {high} years of relevant experience."),
        ("h", "Required Skills"),
        ("p", f"You must be strong in {', '.join(must)}. Knowledge of {', '.join(good)} is a plus."),
        ("h", "Work Mode"),
        ("p", (f"This is a hybrid role. Employees work from the {city} office {days} days a week and from home on the remaining days."
               if remote else f"This is an office based role at our {city} location. Employees cannot work from home on any day.")),
        ("h", "Selection Process"),
        ("p", "The process has an online coding test, two technical interviews and one HR interview."),
    ]
    tag = f"the {role} job at {company}"
    qa = [
        q("Fact", f"How many openings are there for {tag}?", [str(openings)], "easy"),
        q("Fact", f"Which city is {tag} located in?", [city], "easy"),
        q("Fact", f"What experience range is required for {tag}?", [f"{low}", f"{high}"], "medium"),
        q("Fact", f"Which skills are mandatory for {tag}?", must[:2], "medium"),
        q("Yes No", f"Can employees work from home on some days in {tag}?", ["yes" if remote else "no"], "medium"),
        q("Fact", f"How many interview rounds are there for {tag}, other than the coding test?", ["two", "2", "three", "3"] if False else ["two"], "hard"),
        q("Unanswerable", f"What is the salary range for {tag}?", [], "hard", ["salary", "ctc", "lpa"]),
    ]
    return f"Job Description - {role} - {company}", blocks, qa


def resume(rng):
    name = person(rng)
    jobs, total, year = [], 0, 2026
    for company in rng.sample(COMPANIES, 3):
        years = rng.randint(1, 4)
        jobs.append((company, year - years, year, years))
        total += years
        year -= years
    jobs.reverse()
    skills = rng.sample(SKILLS, 6)
    degree, college = rng.choice(DEGREES), rng.choice(COLLEGES)
    cgpa = round(rng.uniform(6.5, 9.4), 1)
    grad = jobs[0][1] - 0
    blocks = [
        ("h", f"Resume - {name}"),
        ("p", f"{name} | {rng.choice(CITIES)}"),
        ("h", "Professional Experience"),
    ] + [("p", f"{c}, {rng.choice(['Analyst', 'Engineer', 'Senior Engineer'])}, {a} to {b} ({y} years).")
         for c, a, b, y in jobs] + [
        ("h", "Skills"),
        ("p", ", ".join(skills) + "."),
        ("h", "Education"),
        ("p", f"{degree}, {college}, CGPA {cgpa}, graduated {grad}."),
        ("h", "Projects"),
        ("p", f"Built a churn prediction pipeline using {skills[0]} and {skills[1]}; reduced manual reporting time by 40 percent."),
    ]
    tag = f"the resume of {name}"
    absent_skill = rng.choice([s for s in SKILLS if s not in skills])
    qa = [
        q("Fact", f"What is the CGPA in {tag}?", [str(cgpa)], "easy"),
        q("Fact", f"Which college did the candidate in {tag} study at?", [college], "easy"),
        q("Fact", f"Which degree does the candidate in {tag} hold?", [degree], "medium"),
        q("Fact", f"In which company did the candidate in {tag} work from {jobs[0][1]} to {jobs[0][2]}?", [jobs[0][0]], "medium"),
        q("Yes No", f"Does the candidate in {tag} list {skills[2]} as a skill?", ["yes"], "medium"),
        q("Yes No", f"Does the candidate in {tag} list {absent_skill} as a skill?", ["no"], "medium"),
        q("Calculation", f"What is the total work experience in years of the candidate in {tag}?", [str(total)], "hard"),
        q("Unanswerable", f"What is the date of birth of the candidate in {tag}?", [], "hard", ["date of birth", "dob", "born"]),
        q("Unanswerable", f"What is the expected salary of the candidate in {tag}?", [], "hard", ["expected salary", "expected ctc"]),
    ]
    return f"Resume - {name}", blocks, qa


def product_manual(rng):
    brand = rng.choice(["Aerofry", "Hydrowash", "Cloudnet", "Brewmate", "Cleanwave"])
    kind, noun = rng.choice([
        ("Air Fryer", "basket"), ("Wi-Fi Router", "antenna"), ("Washing Machine", "drum"),
        ("Coffee Maker", "carafe"), ("Robot Vacuum", "brush"),
    ])
    model = f"{brand} {rng.choice(['X', 'Z', 'Neo'])}{rng.randint(100, 900)}"
    power = rng.randrange(8, 25) * 100
    capacity = rng.choice([2, 3, 4, 5, 6])
    warranty = rng.choice([12, 24, 36])
    codes = rng.sample(range(10, 99), 8)
    problems = [
        ("does not power on", "the power cable is loose; reconnect it firmly and press the power button for 3 seconds"),
        ("overheats", f"the vents are blocked; clean the vents and let the {kind} cool for 20 minutes"),
        ("makes a loud noise", f"a loose {noun}; remove and refit the {noun} until it clicks"),
        ("shows a blinking light", "the self-check failed; unplug for 30 seconds and restart"),
        ("stops in the middle of a cycle", "a safety cutoff was triggered; wait 10 minutes and restart"),
        ("does not respond to the app", "the Wi-Fi signal is weak; move closer to the router and pair again"),
        ("leaks", "the seal is worn out; contact an authorised service centre"),
        ("beeps continuously", "the lid or door is not closed properly; close it fully"),
    ]
    blocks = [("h", f"{model} {kind} - User Manual"),
              ("p", f"Thank you for buying the {model} {kind}. Please read this manual fully before first use.")]
    blocks += [("h", "Safety Instructions")] + [
        ("p", text) for text in [
            f"Keep the {kind} on a flat, dry and stable surface away from water and heat sources.",
            "Do not let children operate the appliance without adult supervision.",
            "Always unplug the appliance before cleaning or moving it.",
            "Do not use the appliance if the cable or plug is damaged.",
            "Use only accessories supplied or approved by the manufacturer.",
        ]]
    blocks += [("h", "Specifications"),
               ("p", f"Rated power: {power} watts. Capacity: {capacity} litres. Warranty: {warranty} months "
                     "from the date of purchase. The warranty does not cover damage from misuse or unauthorised repairs.")]
    blocks += [("h", "Setting Up")] + [
        ("p", f"Step {i}: {step}") for i, step in enumerate([
            "Remove all packaging material and stickers.",
            f"Wash removable parts such as the {noun} with warm soapy water and dry them.",
            "Place the appliance on a level surface with 10 cm free space around it.",
            "Plug the cable into a grounded wall socket.",
            "Press the power button once to complete the first time self-check.",
        ], start=1)]
    blocks += [("h", "Cleaning and Care")] + [
        ("p", text) for text in [
            "Wipe the outside with a soft damp cloth. Never immerse the main body in water.",
            "Clean the filters once a week for best performance.",
            f"Store the {kind} in a cool dry place when not in use for a long time.",
        ]]
    blocks += extra_sections(rng, ["filter care", "descaling", "firmware update", "energy saving", "child lock",
                                   "timer settings", "app pairing", "transport and storage", "recycling", "spare parts"],
                             16, "Guide to")
    blocks += [("h", "Troubleshooting and Error Codes")]
    qa = []
    for code, (problem, fix) in zip(codes, problems):
        blocks.append(("p", f"Error E-{code}: the appliance {problem}. Cause and solution: {fix}."))
    problem_kw = ["power on", "overheat", "noise", "blinking", "middle of a cycle", "app", "leak", "beep"]
    fix_kw = ["reconnect", "20 minutes", "refit", "unplug", "10 minutes", "pair again", "service centre", "close it fully"]
    idx = rng.sample(range(len(problems)), 2)
    tag = f"the {model} {kind}"
    qa = [
        q("Fact", f"What is the rated power of {tag}?", [str(power)], "easy"),
        q("Fact", f"What is the capacity of {tag} in litres?", [str(capacity)], "easy"),
        q("Fact", f"How long is the warranty of {tag}?", [str(warranty)], "easy"),
        q("Fact", f"What does error code E-{codes[idx[0]]} mean for {tag}?", [problem_kw[idx[0]]], "medium"),
        q("Fact", f"What should be done when error E-{codes[idx[1]]} appears on {tag}?", [fix_kw[idx[1]]], "hard"),
        q("Fact", f"How often should the filters of {tag} be cleaned?", ["week"], "medium"),
        q("Yes No", f"Can the main body of {tag} be immersed in water for cleaning?", ["no"], "medium"),
        q("Calculation", f"What is the warranty of {tag} in years?", [str(warranty // 12)], "hard"),
        q("Unanswerable", f"What is the price of {tag}?", [], "hard", ["price", "mrp"]),
        q("Unanswerable", f"What colours is {tag} available in?", [], "hard", ["colour", "color"]),
    ]
    return f"User Manual - {model}", blocks, qa


def api_docs(rng):
    service = f"{rng.choice(['Orbit', 'Nimbus', 'Pulse', 'Atlas', 'Beacon'])} {rng.choice(['Orders', 'Payments', 'Tracking', 'Inventory'])} API"
    base = f"https://api.{service.split()[0].lower()}-{rng.randint(10, 99)}.example.com/v{rng.randint(1, 3)}"
    limit = rng.choice([60, 100, 300, 600])
    header = rng.choice(["X-API-Key", "X-Auth-Token", "Authorization"])
    page = rng.choice([10, 20, 25, 50])
    maxpage = page * rng.choice([2, 4, 10])
    timeout = rng.choice([15, 30, 60])
    resources = rng.sample(["customers", "invoices", "shipments", "products", "refunds", "reports", "webhooks"], 4)
    blocks = [("h", f"{service} Documentation"),
              ("p", f"Base URL: {base}. All requests and responses use JSON."),
              ("h", "Authentication"),
              ("p", f"Send your secret key in the {header} header with every request. Requests without a valid key "
                    "return error 401.")
              ,
              ("h", "Rate Limits"),
              ("p", f"Each key may make {limit} requests per minute. Above this limit the API returns error 429. "
                    "Wait for the number of seconds given in the Retry-After header and try again."),
              ("h", "Pagination"),
              ("p", f"List endpoints return {page} items per page by default. Use the page_size parameter to ask "
                    f"for up to {maxpage} items. Use the cursor parameter to get the next page.")
              ,
              ("h", "Endpoints")]
    for res in resources:
        blocks.append(("p", f"GET /{res} - list all {res}. POST /{res} - create a new item. "
                            f"GET /{res}/{{id}} - fetch one item by its id."))
    blocks += extra_sections(rng, ["webhooks", "idempotency keys", "sandbox mode", "versioning", "filtering",
                                   "sorting", "bulk operations", "audit logs", "field masks", "retries"],
                             14, "Reference for")
    blocks += [("h", "Errors"),
               ("p", "400 means the request is invalid. 401 means authentication failed. 404 means the item was not found. "
                     "429 means too many requests. 500 means a server error; retry after a short delay."),
               ("h", "Timeouts"),
               ("p", f"Requests that take longer than {timeout} seconds are cancelled.")]
    res_a = rng.choice(resources)
    tag = f"the {service}"
    qa = [
        q("Fact", f"What is the rate limit per minute of {tag}?", [str(limit)], "easy"),
        q("Fact", f"Which header carries the secret key in {tag}?", [header], "easy"),
        q("Fact", f"What is the default page size in {tag}?", [str(page)], "medium"),
        q("Fact", f"What is the maximum page size in {tag}?", [str(maxpage)], "medium"),
        q("Fact", f"After how many seconds are long requests cancelled in {tag}?", [str(timeout)], "medium"),
        q("Fact", f"What does error 429 mean in {tag}?", ["too many requests"], "medium"),
        q("Multi", f"What are the rate limit and the request timeout of {tag}?", [str(limit), str(timeout)], "medium"),
        q("Yes No", f"Is there an endpoint to create a new item in /{res_a} in {tag}?", ["yes"], "medium"),
        q("Calculation", f"How many times larger is the maximum page size than the default page size in {tag}?", [str(maxpage // page)], "hard"),
        q("Unanswerable", f"Which programming languages have official SDKs for {tag}?", [], "hard", ["sdk"]),
    ]
    return service, blocks, qa


def annual_report(rng):
    company = rng.choice(COMPANIES)
    fy = rng.choice([2024, 2025])
    segments = rng.sample(["Retail", "Cloud Services", "Manufacturing", "Consulting", "Financial Products", "Exports"], 3)
    seg_rev = [rng.randrange(200, 900) for _ in segments]
    last = [int(v * rng.choice([80, 85, 90, 95]) / 100) for v in seg_rev]
    total, last_total = sum(seg_rev), sum(last)
    profit = int(total * rng.choice([6, 8, 10, 12]) / 100)
    employees = rng.randrange(4, 40) * 100
    dividend = rng.choice([2, 3, 5, 8])
    ceo = person(rng)
    blocks = [
        ("h", f"{company} - Annual Report FY {fy}-{str(fy + 1)[2:]}"),
        ("p", f"Message from the Chief Executive Officer, {ceo}: this year the company grew steadily despite a "
              "difficult economic environment, and we continued to invest in people and technology."),
        ("h", "Financial Highlights"),
        ("p", f"Total revenue for the year was INR {money(total)} crore against INR {money(last_total)} crore "
              f"last year. Net profit was INR {money(profit)} crore. The board recommends a dividend of "
              f"INR {dividend} per share."),
        ("table", ["Segment", "Revenue FY (INR crore)", "Revenue Last FY (INR crore)"],
         [[s, a, b] for s, a, b in zip(segments, seg_rev, last)]),
        ("h", "People"),
        ("p", f"The company had {money(employees)} employees at the end of the year."),
        ("h", "Risk Factors"),
        ("p", "Key risks include changes in regulation, currency movements, dependence on a few large customers "
              "and cyber security threats."),
        ("h", "Outlook"),
        ("p", "We expect moderate growth next year, supported by new product launches and cost discipline."),
    ] + extra_sections(rng, ["corporate governance", "sustainability", "board committees", "related party transactions",
                             "internal controls", "supply chain", "research and development", "community programmes",
                             "auditor remarks", "shareholding pattern"], 20, "Disclosure on", 4)
    seg = rng.randrange(len(segments))
    tag = f"the annual report of {company}"
    growth = round((total - last_total) / last_total * 100)
    qa = [
        q("Fact", f"What was the total revenue in {tag}?", [str(total)], "easy"),
        q("Fact", f"Who is the Chief Executive Officer in {tag}?", [ceo], "easy"),
        q("Fact", f"What dividend per share is recommended in {tag}?", [str(dividend)], "medium"),
        q("Fact", f"How many employees did the company have in {tag}?", [str(employees)], "medium"),
        q("Table Lookup", f"What was the revenue of the {segments[seg]} segment in {tag}?", [str(seg_rev[seg])], "medium"),
        q("Calculation", f"By how many crore did the total revenue increase compared with last year in {tag}?", [str(total - last_total)], "hard"),
        q("Calculation", f"What is the growth in total revenue in percent (rounded to the nearest whole number) in {tag}?", [str(growth)], "hard"),
        q("Yes No", f"Does {tag} mention cyber security as a risk?", ["yes"], "medium"),
        q("Unanswerable", f"What is the share price in {tag}?", [], "hard", ["share price"]),
    ]
    return f"Annual Report - {company}", blocks, qa


def meeting_transcript(rng):
    project = f"Project {rng.choice(['Aurora', 'Falcon', 'Helios', 'Mango', 'Quartz', 'Tundra'])}"
    people = rng.sample([f"{f}" for f in FIRST], 4)
    budget = rng.randrange(5, 40) * 100000
    deadline = date(rng, 2026)
    vendor = rng.choice(["Skyline Cloud", "DataForge", "Pixel Works", "NetCore"])
    tool = rng.choice(["Jira", "Notion", "Linear", "Asana"])
    users = rng.randrange(2, 20) * 100
    owner, other = people[1], people[2]
    lines = [
        f"{people[0]}: Good morning everyone. Let us review {project}. First, where are we on the schedule?",
        f"{people[1]}: Development is on track. The first release is planned for {deadline}.",
        f"{people[2]}: Testing needs more time. We found a few issues in the payment flow, but nothing serious.",
        f"{people[0]}: Okay. What about the budget?",
        f"{people[3]}: The approved budget is INR {money(budget)}. So far we have used about half of it.",
        f"{people[0]}: Good. We also need a vendor for hosting. Any suggestions?",
        f"{people[3]}: I compared three options. I recommend {vendor} because of pricing and support.",
        f"{people[1]}: I agree with {vendor}. Let us finalise it today.",
        f"{people[0]}: Agreed. {people[1]} will own the vendor setup. {people[2]} will share the test report by Friday.",
        f"{people[2]}: Understood. One more thing, we expect around {money(users)} users in the first month.",
        f"{people[0]}: Noted. We will track all tasks in {tool} from now on.",
        f"{people[3]}: One risk is staff leave during the festival week. We may need a backup person.",
        f"{people[0]}: Let us handle that offline. Thanks everyone, meeting closed.",
    ]
    chatter = [
        "Let me share my screen. Can everyone see the dashboard?",
        "I think we should keep the next sprint small and finish what we started.",
        "That makes sense. Let us not add new features before the release.",
        "Can we move the design review to the afternoon? I have a clash.",
        "Sure, I will update the calendar invite after this call.",
        "We should write down the open questions so that nothing gets lost.",
        "I will take that as an action item and follow up by email.",
        "The customer feedback so far has been mostly positive, with a few requests for dark mode.",
        "Let us keep the documentation updated as we go, not at the end.",
        "I will check with the support team about the common complaints.",
    ]
    middle = [("p", f"{rng.choice(people)}: {rng.choice(chatter)}") for _ in range(45)]
    blocks = [("h", f"Transcript - {project} Review Meeting")] + [("p", line) for line in lines[:6]] + \
             middle[:22] + [("p", line) for line in lines[6:]] + middle[22:]
    tag = f"the {project} review meeting"
    qa = [
        q("Fact", f"Which vendor was chosen for hosting in {tag}?", [vendor], "easy"),
        q("Fact", f"What is the approved budget in {tag}?", [str(budget)], "easy"),
        q("Fact", f"When is the first release planned in {tag}?", [deadline], "medium"),
        q("Fact", f"Who will own the vendor setup in {tag}?", [owner], "medium"),
        q("Fact", f"Which tool will be used to track tasks in {tag}?", [tool], "medium"),
        q("Fact", f"How many users are expected in the first month in {tag}?", [str(users)], "medium"),
        q("Fact", f"Who will share the test report in {tag}?", [other], "medium"),
        q("Yes No", f"Did the participants agree on {vendor} in {tag}?", ["yes"], "medium"),
        q("Unanswerable", f"Who was absent from {tag}?", [], "hard", ["absent"]),
        q("Unanswerable", f"What was the venue of {tag}?", [], "hard", ["venue"]),
    ]
    return f"Meeting Transcript - {project}", blocks, qa


def support_faq(rng):
    store = rng.choice(["Cartly", "Shopnest", "Dealbazaar", "Kartwise", "Bagmart"])
    ret = rng.choice([7, 10, 15, 30])
    refund = rng.choice([3, 5, 7])
    free_ship = rng.choice([499, 699, 999])
    ship_fee = rng.choice([40, 49, 59])
    cod = rng.choice([True, False])
    hours = rng.choice(["9 am to 6 pm", "10 am to 7 pm", "9 am to 9 pm"])
    warranty = rng.choice([6, 12])
    blocks = [("h", f"{store} - Customer Support FAQ"),
              ("h", "How long do I have to return a product?"),
              ("p", f"You can return most products within {ret} days of delivery. The product must be unused and in original packaging."),
              ("h", "When will I get my refund?"),
              ("p", f"Refunds are processed within {refund} business days after we receive the returned product, to your original payment method."),
              ("h", "What are the delivery charges?"),
              ("p", f"Delivery is free on orders above INR {free_ship}. For smaller orders, a delivery fee of INR {ship_fee} applies."),
              ("h", "Is cash on delivery available?"),
              ("p", "Yes, cash on delivery is available on orders up to INR 5,000." if cod
               else "No, cash on delivery is not available. Please pay online by card, UPI or net banking."),
              ("h", "How can I contact support?"),
              ("p", f"Our support team is available from {hours}, Monday to Saturday, through chat and email."),
              ("h", "Do products have a warranty?"),
              ("p", f"Electronics carry a manufacturer warranty of {warranty} months. Clothing and books have no warranty."),
              ("h", "Can I cancel my order?"),
              ("p", "Orders can be cancelled before they are shipped from the My Orders page.")]
    tag = f"the {store} FAQ"
    qa = [
        q("Fact", f"Within how many days can products be returned according to {tag}?", [str(ret)], "easy"),
        q("Fact", f"How many business days does a refund take according to {tag}?", [str(refund)], "easy"),
        q("Fact", f"Above what order value is delivery free according to {tag}?", [str(free_ship)], "medium"),
        q("Fact", f"What is the delivery fee for small orders according to {tag}?", [str(ship_fee)], "medium"),
        q("Fact", f"What are the support hours according to {tag}?", [hours.split(" to ")[0]], "medium"),
        q("Fact", f"How long is the warranty on electronics according to {tag}?", [str(warranty)], "medium"),
        q("Yes No", f"Is cash on delivery available according to {tag}?", ["yes" if cod else "no"], "medium"),
        q("Yes No", f"Can an order be cancelled before it is shipped according to {tag}?", ["yes"], "medium"),
        q("Unanswerable", f"Does {store} deliver outside India according to {tag}?", [], "hard", ["international", "outside india"]),
    ]
    return f"Customer FAQ - {store}", blocks, qa


def govt_circular(rng):
    dept = rng.choice(["Department of Skill Development", "Ministry of Rural Livelihood",
                       "Department of Women and Child Welfare", "Directorate of Technical Education"])
    scheme = f"{rng.choice(['Yuva', 'Gramin', 'Nari', 'Udyam', 'Vidya'])} {rng.choice(['Sahay', 'Shakti', 'Kaushal', 'Utthan'])} Yojana"
    number = f"No. {rng.randint(10, 99)}/{rng.randint(2024, 2026)}/{rng.choice(['SD', 'RL', 'WC', 'TE'])}"
    amin, amax = rng.choice([(18, 25), (18, 30), (21, 35)])
    income = rng.randrange(2, 8) * 100000
    benefit = rng.randrange(2, 10) * 500
    months = rng.choice([6, 12])
    last_date = date(rng, 2026)
    helpline = f"1800-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    docs = rng.sample(["Aadhaar card", "income certificate", "bank passbook", "educational certificates",
                       "caste certificate", "passport size photograph"], 3)
    blocks = [
        ("h", f"Government of the State - {dept}"),
        ("p", f"Circular {number}. Subject: Applications invited under the {scheme}."),
        ("h", "Eligibility"),
        ("p", f"Applicants must be between {amin} and {amax} years of age. The annual family income must not "
              f"exceed INR {money(income)}."),
        ("h", "Benefit"),
        ("p", f"Selected beneficiaries will receive INR {money(benefit)} per month for {months} months, "
              "transferred directly to their bank account."),
        ("h", "How to Apply"),
        ("p", f"Applications must be submitted online on the department portal on or before {last_date}. "
              f"Please keep these documents ready: {', '.join(docs)}."),
        ("h", "Contact"),
        ("p", f"For help, call the toll free helpline {helpline} between 10 am and 5 pm on working days."),
    ] + extra_sections(rng, ["verification of applications", "selection committee", "grievance handling",
                             "monitoring and audit", "disbursement schedule"], 6, "Instruction on")
    tag = f"the {scheme} circular"
    qa = [
        q("Fact", f"What is the last date to apply under {tag}?", [last_date], "easy"),
        q("Fact", f"What is the maximum annual family income allowed under {tag}?", [str(income)], "easy"),
        q("Fact", f"What is the monthly benefit under {tag}?", [str(benefit)], "easy"),
        q("Fact", f"What is the helpline number in {tag}?", [helpline], "medium"),
        q("Fact", f"What age range is eligible under {tag}?", [str(amin), str(amax)], "medium"),
        q("Fact", f"Which documents are required under {tag}?", docs[:2], "medium"),
        q("Calculation", f"What is the total benefit a beneficiary receives under {tag}?", [str(benefit * months)], "hard"),
        q("Yes No", f"Is the benefit transferred directly to the bank account under {tag}?", ["yes"], "medium"),
        q("Unanswerable", f"Can foreign nationals apply under {tag}?", [], "hard", ["foreign"]),
    ]
    return f"Circular - {scheme}", blocks, qa


# (name, builder, formats, extra documents for near-duplicate / confusing types)
BUILDERS = [
    ("rent_agreement", rent_agreement, ["docx", "pdf"], 1),
    ("offer_letter", offer_letter, ["pdf", "docx"], 1),
    ("insurance_policy", insurance_policy, ["pdf"], 0),
    ("emi_schedule", emi_schedule, ["csv"], 0),
    ("sales_report", sales_report, ["csv"], 0),
    ("syllabus", syllabus, ["docx", "pdf"], 0),
    ("job_description", job_description, ["txt", "docx"], 0),
    ("resume", resume, ["pdf", "docx"], 0),
    ("product_manual", product_manual, ["pdf"], 0),
    ("api_docs", api_docs, ["txt"], 0),
    ("annual_report", annual_report, ["pdf"], 0),
    ("meeting_transcript", meeting_transcript, ["txt"], 0),
    ("support_faq", support_faq, ["docx", "txt"], 0),
    ("govt_circular", govt_circular, ["pdf"], 0),
]

STYLES = ["Plain", "Polite", "Hinglish"]


def style_question(question, style):
    if style == "Polite":
        return f"Could you please tell me: {question}"
    if style == "Hinglish":
        return f"{question} (thoda short me batao)"
    return question


def build_corpus(seed: int = 42, docs_per_type: int = 3) -> list:
    """Return documents: type, format, title, blocks, questions, file name."""
    rng = random.Random(seed)
    corpus = []
    used_titles = set()   # no two documents may share a name (keeps answers unambiguous)

    for type_name, builder, formats, extra in BUILDERS:
        for number in range(docs_per_type + extra):
            for _ in range(50):
                title, blocks, qa = builder(rng)
                if title not in used_titles:
                    break
            used_titles.add(title)
            file_format = formats[number % len(formats)]
            questions = []
            for item in qa:
                style = rng.choices(STYLES, weights=[6, 2, 2])[0]
                questions.append({
                    **item,
                    "question": style_question(item["question"], style),
                    "style": style,
                })
            corpus.append({
                "doc_type": type_name,
                "format": file_format,
                "title": title,
                "blocks": blocks,
                "questions": questions,
                "file_name": f"{type_name}_{number + 1:02d}.{file_format}",
            })

    return corpus