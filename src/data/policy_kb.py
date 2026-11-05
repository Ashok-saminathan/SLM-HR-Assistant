"""Synthetic HR policy knowledge base for the fictional company "Nimbus Works".

All content below is invented for this project. It contains no real company data
and no personal data. Each policy has:
  id      - stable identifier, also used as the citation "source" in answers
  title   - short title
  text    - the policy excerpt that is shown to the model as context
  qa      - list of (question, answer, key_facts). key_facts are short strings that
            must appear in a correct answer (used by the evaluation metrics).
"""

COMPANY = "Nimbus Works"

REFUSAL_ANSWER = (
    "I could not find this in the HR policy excerpt provided. "
    "Please contact the HR helpdesk for help."
)

POLICIES = [
    {
        "id": "leave-annual",
        "title": "Annual Leave",
        "text": "Full-time employees receive 18 days of paid annual leave per calendar year, accrued at 1.5 days per month. Up to 10 unused days can be carried forward to the next year. Any remaining unused days lapse on 31 March.",
        "qa": [
            ("How many days of annual leave do I get each year?", "Full-time employees get 18 days of paid annual leave per calendar year.", ["18"]),
            ("How many unused leave days can I carry forward?", "Up to 10 unused annual leave days can be carried forward to the next year.", ["10"]),
            ("When do my leftover leave days expire?", "Unused days beyond the carry-forward limit lapse on 31 March.", ["31 March"]),
            ("At what rate does annual leave accrue?", "Annual leave accrues at 1.5 days per month.", ["1.5"]),
        ],
    },
    {
        "id": "leave-sick",
        "title": "Sick Leave",
        "text": "Employees get 12 paid sick days per year. A medical certificate is required when the absence is longer than 2 consecutive days. Employees must inform their manager before 10:00 AM on the first day of absence.",
        "qa": [
            ("How many sick days are allowed per year?", "Employees get 12 paid sick days per year.", ["12"]),
            ("When do I need a medical certificate for sick leave?", "A medical certificate is required if the absence is longer than 2 consecutive days.", ["2 consecutive"]),
            ("By what time should I inform my manager that I am sick?", "You must inform your manager before 10:00 AM on the first day of absence.", ["10:00 AM"]),
            ("Do I need a doctor's note for a one-day sick absence?", "No. A medical certificate is only required when the absence is longer than 2 consecutive days.", ["2 consecutive"]),
        ],
    },
    {
        "id": "leave-casual",
        "title": "Casual Leave",
        "text": "Each employee gets 6 days of casual leave per year for short personal needs. A maximum of 2 consecutive days can be taken at a time. Casual leave cannot be combined with annual leave or sick leave.",
        "qa": [
            ("How many casual leave days do employees get?", "Employees get 6 days of casual leave per year.", ["6"]),
            ("What is the maximum number of consecutive casual leave days?", "A maximum of 2 consecutive days can be taken at a time.", ["2"]),
            ("Can I attach casual leave to my annual leave?", "No. Casual leave cannot be combined with annual leave or sick leave.", ["cannot"]),
        ],
    },
    {
        "id": "leave-maternity",
        "title": "Maternity Leave",
        "text": "Eligible employees get 26 weeks of paid maternity leave for the first two children and 12 weeks from the third child onward. Employees should give written notice at least 8 weeks before the expected delivery date.",
        "qa": [
            ("How long is paid maternity leave for a first child?", "Paid maternity leave is 26 weeks for the first two children.", ["26 weeks"]),
            ("How much maternity leave do I get for my third child?", "For the third child onward, maternity leave is 12 weeks.", ["12 weeks"]),
            ("How early must I tell HR about my maternity leave?", "Written notice must be given at least 8 weeks before the expected delivery date.", ["8 weeks"]),
        ],
    },
    {
        "id": "leave-paternity",
        "title": "Paternity Leave",
        "text": "Employees who become fathers are entitled to 10 working days of paid paternity leave. The leave must be taken within 3 months of the child's birth.",
        "qa": [
            ("How many days of paternity leave are available?", "Employees are entitled to 10 working days of paid paternity leave.", ["10"]),
            ("Within what time must paternity leave be used?", "It must be taken within 3 months of the child's birth.", ["3 months"]),
            ("Is paternity leave paid?", "Yes, paternity leave of 10 working days is paid.", ["paid"]),
        ],
    },
    {
        "id": "leave-bereavement",
        "title": "Bereavement Leave",
        "text": "Employees get 5 paid days of bereavement leave on the death of an immediate family member (spouse, child, parent or sibling). For grandparents and parents-in-law, 2 paid days are given.",
        "qa": [
            ("How many bereavement leave days do I get if my parent passes away?", "You get 5 paid days of bereavement leave for the death of a parent.", ["5"]),
            ("How much leave is given when a grandparent dies?", "For grandparents, 2 paid days of bereavement leave are given.", ["2"]),
            ("Which relatives count as immediate family for bereavement leave?", "Immediate family means spouse, child, parent or sibling.", ["spouse", "sibling"]),
        ],
    },
    {
        "id": "leave-marriage",
        "title": "Marriage Leave",
        "text": "Employees are eligible for 5 days of paid marriage leave, available once during their employment. The leave request must be submitted at least 14 days in advance.",
        "qa": [
            ("How many days of marriage leave can I take?", "Employees can take 5 days of paid marriage leave.", ["5"]),
            ("Can I take marriage leave more than once?", "No. Marriage leave is available only once during employment.", ["once"]),
            ("How far in advance should I apply for marriage leave?", "The request must be submitted at least 14 days in advance.", ["14 days"]),
        ],
    },
    {
        "id": "work-hours",
        "title": "Working Hours",
        "text": "Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday, including a 1-hour lunch break. Core hours are 11:00 AM to 4:00 PM, when all employees must be available for meetings.",
        "qa": [
            ("What are the standard office hours?", "Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday.", ["9:30 AM", "6:30 PM"]),
            ("How long is the lunch break?", "The lunch break is 1 hour and is included in the working hours.", ["1 hour"]),
            ("What are the core hours?", "Core hours are 11:00 AM to 4:00 PM.", ["11:00 AM", "4:00 PM"]),
            ("Do we work on Saturdays?", "No. Standard working days are Monday to Friday.", ["Monday to Friday"]),
        ],
    },
    {
        "id": "wfh",
        "title": "Work From Home",
        "text": "Under the hybrid policy, employees may work from home up to 2 days per week with manager approval. Work-from-home days must be requested at least 1 day in advance. Employees on probation are not eligible for work from home.",
        "qa": [
            ("How many days a week can I work from home?", "You can work from home up to 2 days per week with manager approval.", ["2 days"]),
            ("How early do I need to request a work-from-home day?", "Requests must be made at least 1 day in advance.", ["1 day"]),
            ("Can employees on probation work from home?", "No. Employees on probation are not eligible for work from home.", ["not eligible"]),
        ],
    },
    {
        "id": "overtime",
        "title": "Overtime",
        "text": "Overtime must be approved by the manager in advance. It is paid at 1.5 times the hourly rate on weekdays and 2 times on weekends and public holidays. Overtime pay applies only to grades L1 to L3. Compensatory time off can be taken instead within 30 days.",
        "qa": [
            ("How much is overtime paid on weekdays?", "Weekday overtime is paid at 1.5 times the hourly rate.", ["1.5"]),
            ("What is the overtime rate on public holidays?", "Overtime on weekends and public holidays is paid at 2 times the hourly rate.", ["2 times"]),
            ("Which grades are eligible for overtime pay?", "Overtime pay applies only to grades L1 to L3.", ["L1 to L3"]),
            ("Can I take time off instead of overtime pay?", "Yes, compensatory time off can be taken within 30 days.", ["30 days"]),
        ],
    },
    {
        "id": "probation",
        "title": "Probation",
        "text": "New employees serve a probation period of 6 months. The notice period during probation is 15 days. A confirmation letter is issued within 2 weeks after a successful probation review.",
        "qa": [
            ("How long is the probation period?", "The probation period is 6 months.", ["6 months"]),
            ("What is the notice period during probation?", "The notice period during probation is 15 days.", ["15 days"]),
            ("When will I get my confirmation letter?", "The confirmation letter is issued within 2 weeks after a successful probation review.", ["2 weeks"]),
        ],
    },
    {
        "id": "notice-period",
        "title": "Notice Period",
        "text": "After confirmation, the notice period is 60 days for grades L1 to L4 and 90 days for grade L5 and above. Employees may buy out the shortfall in notice by paying basic salary for the remaining days, subject to manager and HR approval.",
        "qa": [
            ("What is the notice period for a confirmed L3 employee?", "The notice period for grades L1 to L4 is 60 days.", ["60 days"]),
            ("How long is the notice period for L5 and above?", "For grade L5 and above, the notice period is 90 days.", ["90 days"]),
            ("Can I buy out my notice period?", "Yes. You can buy out the shortfall by paying basic salary for the remaining days, subject to manager and HR approval.", ["basic salary"]),
        ],
    },
    {
        "id": "salary-payment",
        "title": "Salary Payment",
        "text": "Salaries are credited on the last working day of each month. Payslips are available on the HR portal within 2 working days after the salary credit.",
        "qa": [
            ("When is salary credited?", "Salaries are credited on the last working day of each month.", ["last working day"]),
            ("Where can I download my payslip?", "Payslips are available on the HR portal.", ["HR portal"]),
            ("How soon after payment is the payslip available?", "Payslips are available within 2 working days after the salary credit.", ["2 working days"]),
        ],
    },
    {
        "id": "travel-expenses",
        "title": "Travel Expenses",
        "text": "For domestic business travel, employees may book economy air tickets or AC 3-tier train tickets. Hotel stays are reimbursed up to INR 4,500 per night in metro cities and INR 3,000 per night elsewhere. Expense claims must be submitted within 15 days of returning from the trip.",
        "qa": [
            ("What is the hotel limit per night in metro cities?", "The hotel limit is INR 4,500 per night in metro cities.", ["4,500"]),
            ("What is the hotel limit in non-metro cities?", "In other cities, hotel stays are reimbursed up to INR 3,000 per night.", ["3,000"]),
            ("How long do I have to submit my travel expense claim?", "Claims must be submitted within 15 days of returning from the trip.", ["15 days"]),
            ("Which class of train travel is allowed?", "Employees may book AC 3-tier train tickets.", ["AC 3-tier"]),
        ],
    },
    {
        "id": "meal-allowance",
        "title": "Meal Allowance",
        "text": "Employees on business travel receive a meal allowance of INR 800 per day. Client entertainment expenses need prior written approval from the department head.",
        "qa": [
            ("How much is the daily meal allowance during travel?", "The meal allowance is INR 800 per day during business travel.", ["800"]),
            ("Do I need approval to entertain a client?", "Yes. Client entertainment needs prior written approval from the department head.", ["department head"]),
            ("Who approves client entertainment expenses?", "The department head must approve them in writing beforehand.", ["department head"]),
        ],
    },
    {
        "id": "health-insurance",
        "title": "Group Health Insurance",
        "text": "The group health insurance covers the employee, spouse and up to 2 children with a sum insured of INR 5 lakh. Parents can be added, but the additional premium is paid by the employee. Coverage starts from the date of joining.",
        "qa": [
            ("What is the sum insured in the group health insurance?", "The sum insured is INR 5 lakh.", ["5 lakh"]),
            ("Who is covered under the health insurance?", "It covers the employee, spouse and up to 2 children.", ["spouse", "2 children"]),
            ("Can I add my parents to the insurance?", "Yes, but the additional premium for parents is paid by the employee.", ["premium"]),
            ("When does insurance coverage begin?", "Coverage starts from the date of joining.", ["date of joining"]),
        ],
    },
    {
        "id": "provident-fund",
        "title": "Provident Fund",
        "text": "The employer contributes 12% of basic salary to the Provident Fund, matched by an employee contribution of 12% of basic salary. Contributions are deducted monthly and shown on the payslip.",
        "qa": [
            ("How much does the company contribute to PF?", "The employer contributes 12% of basic salary to the Provident Fund.", ["12%"]),
            ("What percentage of my salary goes to PF?", "The employee contribution is 12% of basic salary.", ["12%"]),
            ("Where can I see my PF deduction?", "The deduction is shown on the payslip.", ["payslip"]),
        ],
    },
    {
        "id": "gratuity",
        "title": "Gratuity",
        "text": "Gratuity is payable after 5 years of continuous service. It is calculated as 15 days of basic salary for each completed year of service.",
        "qa": [
            ("After how many years am I eligible for gratuity?", "Gratuity is payable after 5 years of continuous service.", ["5 years"]),
            ("How is gratuity calculated?", "It is 15 days of basic salary for each completed year of service.", ["15 days"]),
            ("Am I eligible for gratuity after 3 years?", "No. Gratuity is payable only after 5 years of continuous service.", ["5 years"]),
        ],
    },
    {
        "id": "annual-bonus",
        "title": "Annual Bonus",
        "text": "The annual performance bonus is paid in April and ranges from 5% to 15% of annual fixed pay depending on the performance rating. Employees must be on the company rolls on 31 March to be eligible.",
        "qa": [
            ("In which month is the annual bonus paid?", "The annual bonus is paid in April.", ["April"]),
            ("What is the range of the annual bonus?", "The bonus ranges from 5% to 15% of annual fixed pay.", ["5%", "15%"]),
            ("What is the eligibility date for the bonus?", "Employees must be on the company rolls on 31 March.", ["31 March"]),
        ],
    },
    {
        "id": "performance-review",
        "title": "Performance Review",
        "text": "Performance reviews take place twice a year, in April and October. Employees are rated on a 5-point scale. A rating of 1 leads to a 60-day performance improvement plan.",
        "qa": [
            ("How often are performance reviews held?", "Performance reviews are held twice a year, in April and October.", ["twice"]),
            ("What rating scale is used?", "Employees are rated on a 5-point scale.", ["5-point"]),
            ("What happens if I get a rating of 1?", "A rating of 1 leads to a 60-day performance improvement plan.", ["60-day"]),
        ],
    },
    {
        "id": "learning-budget",
        "title": "Learning Budget",
        "text": "Every employee has a learning budget of INR 25,000 per year for courses and certifications. If an employee leaves within 12 months of a reimbursement, the amount must be repaid.",
        "qa": [
            ("How much is the yearly learning budget?", "The learning budget is INR 25,000 per year.", ["25,000"]),
            ("What can the learning budget be used for?", "It can be used for courses and certifications.", ["certifications"]),
            ("What if I resign soon after claiming a course fee?", "If you leave within 12 months of a reimbursement, the amount must be repaid.", ["12 months"]),
        ],
    },
    {
        "id": "dress-code",
        "title": "Dress Code",
        "text": "The dress code is business casual from Monday to Thursday and casual on Fridays. Formal attire is required on days when employees visit client sites.",
        "qa": [
            ("What is the dress code on Fridays?", "The dress code on Fridays is casual.", ["casual"]),
            ("What should I wear Monday to Thursday?", "Business casual is the dress code from Monday to Thursday.", ["business casual"]),
            ("Is formal wear needed for client visits?", "Yes. Formal attire is required on client-site days.", ["formal"]),
        ],
    },
    {
        "id": "anti-harassment",
        "title": "Anti-Harassment",
        "text": "Nimbus Works has zero tolerance for harassment of any kind. Complaints can be sent to the Internal Committee at ic@nimbusworks.example. The committee completes its inquiry within 30 days of receiving the complaint.",
        "qa": [
            ("Where do I report harassment?", "Complaints can be sent to the Internal Committee at ic@nimbusworks.example.", ["Internal Committee"]),
            ("How long does the harassment inquiry take?", "The committee completes its inquiry within 30 days of receiving the complaint.", ["30 days"]),
            ("What is the company stance on harassment?", "The company has zero tolerance for harassment of any kind.", ["zero tolerance"]),
        ],
    },
    {
        "id": "grievance",
        "title": "Grievance Handling",
        "text": "Employees can raise a grievance with their HR Business Partner. HR acknowledges the grievance within 2 working days and aims to resolve it within 15 working days.",
        "qa": [
            ("Who should I approach with a workplace grievance?", "You can raise it with your HR Business Partner.", ["HR Business Partner"]),
            ("How quickly will HR acknowledge my grievance?", "HR acknowledges a grievance within 2 working days.", ["2 working days"]),
            ("What is the target time to resolve a grievance?", "HR aims to resolve grievances within 15 working days.", ["15 working days"]),
        ],
    },
    {
        "id": "referral-bonus",
        "title": "Employee Referral Bonus",
        "text": "Employees receive a referral bonus of INR 30,000 when a referred candidate completes 90 days after joining. The bonus is INR 50,000 for referrals to roles at grade L5 and above.",
        "qa": [
            ("How much is the referral bonus?", "The referral bonus is INR 30,000.", ["30,000"]),
            ("When is the referral bonus paid?", "It is paid when the referred candidate completes 90 days after joining.", ["90 days"]),
            ("What is the referral bonus for senior roles?", "For roles at grade L5 and above, the bonus is INR 50,000.", ["50,000"]),
        ],
    },
    {
        "id": "public-holidays",
        "title": "Public Holidays",
        "text": "Nimbus Works observes 12 paid public holidays per year, and the list is published in January. Employees can also choose 2 optional holidays from a list of 8.",
        "qa": [
            ("How many public holidays are there in a year?", "There are 12 paid public holidays per year.", ["12"]),
            ("When is the holiday list published?", "The holiday list is published in January.", ["January"]),
            ("How many optional holidays can I choose?", "You can choose 2 optional holidays from a list of 8.", ["2 optional"]),
        ],
    },
    {
        "id": "exit-process",
        "title": "Exit Process",
        "text": "On the last working day, employees must return the laptop and ID card. Full and final settlement is processed within 45 days of the last working day.",
        "qa": [
            ("What do I need to return when I leave?", "You must return the laptop and ID card on your last working day.", ["laptop", "ID card"]),
            ("When will I receive my full and final settlement?", "It is processed within 45 days of the last working day.", ["45 days"]),
            ("On which day do I hand over company assets?", "Assets must be returned on the last working day.", ["last working day"]),
        ],
    },
    {
        "id": "attendance",
        "title": "Attendance and Late Arrival",
        "text": "Attendance is recorded through the mobile app punch-in. Arriving after 9:45 AM counts as a late arrival. Three late arrivals in a calendar month are treated as a half-day leave deduction.",
        "qa": [
            ("After what time am I marked late?", "Arriving after 9:45 AM counts as a late arrival.", ["9:45 AM"]),
            ("What happens if I am late three times a month?", "Three late arrivals in a month are treated as a half-day leave deduction.", ["half-day"]),
            ("How is attendance recorded?", "Attendance is recorded through the mobile app punch-in.", ["mobile app"]),
        ],
    },
    {
        "id": "data-security",
        "title": "Device and Data Security",
        "text": "Company laptops must use full-disk encryption. Personal USB drives are not allowed on company devices. A lost or stolen device must be reported to the IT helpdesk within 2 hours.",
        "qa": [
            ("Can I plug a personal USB drive into my work laptop?", "No. Personal USB drives are not allowed on company devices.", ["not allowed"]),
            ("How soon must I report a lost laptop?", "A lost or stolen device must be reported to the IT helpdesk within 2 hours.", ["2 hours"]),
            ("What encryption is required on laptops?", "Company laptops must use full-disk encryption.", ["full-disk"]),
        ],
    },
    {
        "id": "relocation",
        "title": "Relocation Assistance",
        "text": "Employees transferred to another city by the company receive a lump-sum relocation allowance of INR 60,000. The employee must stay with the company for at least 18 months after relocating, otherwise the allowance is recovered.",
        "qa": [
            ("How much relocation allowance is given?", "A lump-sum relocation allowance of INR 60,000 is given.", ["60,000"]),
            ("How long must I stay after relocating?", "You must stay at least 18 months after relocating.", ["18 months"]),
            ("What if I leave before 18 months after a transfer?", "The relocation allowance is recovered.", ["recovered"]),
        ],
    },
    {
        "id": "internal-transfer",
        "title": "Internal Transfer",
        "text": "Employees can apply for an internal job transfer after completing 12 months in their current role. The application needs approval from the current manager and the hiring manager.",
        "qa": [
            ("When can I apply for an internal transfer?", "After completing 12 months in your current role.", ["12 months"]),
            ("Whose approval is needed for an internal transfer?", "You need approval from your current manager and the hiring manager.", ["current manager", "hiring manager"]),
            ("Can I apply for a transfer in my 6th month?", "No. You must complete 12 months in the current role first.", ["12 months"]),
        ],
    },
    {
        "id": "employee-assistance",
        "title": "Employee Assistance Programme",
        "text": "The Employee Assistance Programme offers free and confidential counselling, with up to 6 sessions per employee per year. Sessions can be booked through the HR portal and are also available to immediate family members.",
        "qa": [
            ("How many free counselling sessions do I get?", "You get up to 6 free counselling sessions per year.", ["6"]),
            ("How do I book a counselling session?", "Sessions can be booked through the HR portal.", ["HR portal"]),
            ("Can my family use the counselling service?", "Yes. It is also available to immediate family members.", ["family"]),
        ],
    },
    {
        "id": "joining-formalities",
        "title": "Joining Formalities",
        "text": "New joiners must submit their ID proof, address proof and educational certificates on or before the date of joining. Background verification is completed within 30 days. The laptop and access card are issued on the first day.",
        "qa": [
            ("What documents must I submit when joining?", "You must submit ID proof, address proof and educational certificates.", ["ID proof", "address proof"]),
            ("How long does background verification take?", "Background verification is completed within 30 days.", ["30 days"]),
            ("When do I receive my laptop?", "The laptop and access card are issued on the first day.", ["first day"]),
        ],
    },
    {
        "id": "parking-transport",
        "title": "Parking and Transport",
        "text": "Free two-wheeler parking is available at all offices. Four-wheeler parking costs INR 1,500 per month, deducted from salary. A shuttle service runs from the metro station between 8:30 AM and 10:00 AM and from 5:30 PM to 7:30 PM.",
        "qa": [
            ("Is two-wheeler parking free?", "Yes, two-wheeler parking is free at all offices.", ["free"]),
            ("How much does car parking cost?", "Four-wheeler parking costs INR 1,500 per month.", ["1,500"]),
            ("When does the office shuttle run in the morning?", "The shuttle runs from the metro station between 8:30 AM and 10:00 AM.", ["8:30 AM", "10:00 AM"]),
        ],
    },
]

# Questions the HR policy excerpts can never answer (used for the "refuse" behaviour).
OUT_OF_SCOPE_QUESTIONS = [
    "What is the CEO's monthly salary?",
    "Which stocks should I invest in this year?",
    "What is the weather in Chennai today?",
    "Can you recommend a good restaurant near the office?",
    "How many employees work at Nimbus Works?",
    "What is the company's revenue for last quarter?",
    "Who won the cricket match yesterday?",
    "How do I reset my personal email password?",
    "Will there be layoffs next year?",
    "What is the Wi-Fi password in the cafeteria?",
    "How do I file my personal income tax return?",
    "Is the company planning to open an office in Berlin?",
    "What is the stock option vesting schedule for founders?",
    "Can I bring my dog to the office?",
    "What time does the gym on the 5th floor open?",
    "How much is the maternity leave policy of other companies?",
    "Write a poem about Mondays.",
    "What is the name of my reporting manager?",
    "What is my current salary?",
    "Who is the CFO of the company?",
    "Which laptop model will I get?",
    "Is there a free lunch on Wednesdays?",
    "How many casual leave days are left in my account?",
    "Translate the leave policy into French.",
    "What is the sabbatical policy for senior leaders?",
]
