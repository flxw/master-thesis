# BPIC 2011
We have prepared a real-life log, taken from a Dutch Academic Hospital. This log contains some 150.000 events in over 1100 cases. Apart from some anonymization, the log contains all data as it came from the Hospital's systems. Each case is a patient of a Gynaecology department. The log contains information about when certain activities took place, which group performed the activity and so on. Many attributes have been recorded that are relevant to the process. Some attributes are repeated more than once for a patient, indicating that this patient went through different (maybe overlapping) phases, where a phase consists of the combination Diagnosis & Treatment.

# BPIC 2012
We have prepared a real-life log, taken from a Dutch Financial Institute. This log contains some 262.200 events in 13.087 cases. Apart from some anonymization, the log contains all data as it came from the financial institute. The process represented in the event log is an application process for a personal loan or overdraft within a global financing organization. The amount requested by the customer is indicated in the case attribute AMOUNT_REQ, which is global, i.e. every case contains this attribute. The event log is a merger of three intertwined sub processes. The first letter of each task name identifies from which sub process (source) it originated from. Feel free to run analyses on the process as a whole, on selections of the whole process and/or the individual sub processes.

# BPIC 2016
We are proud to say that the data for this year's BPI challenge is provided by the same company as BPI 2012. The same process is considered, five years later!

The dataset provided by the company this time is richer than before. An important difference is that the company switched systems and now supports multiple offers for a single application (in contrast to 2012, where a work-around was clearly visible in the log).

The event log provided this year contains all applications filed in 2016, and their subsequent handling up to February 2nd 2017. In total, there are 1,202,267 events pertaining to 31,509 loan applications. For these applications, a total of 42,995 offers were created. As in 2012, we have three types of events, namely Application state changes, Offer state changes and Workflow events. There are 149 originators in the data, i.e. employees or systems of the company.

For all applications, the following data is available:

Requested load amount (in Euro),
The application type,
The reason the loan was applied for (LoanGoal), and
An application ID.
For all offers, the following data is available:

* An offer ID,
* The offered amount,
* The initial withdrawal amount,
* The number of payback terms agreed to,
* The monthly costs,
* The creditscore of the customer,
* The employee who created the offer,
* Whether the offer was selected, and
* Whether the offer was accepted by the customer

Next to this information, many events are recorded in the log. For each (uniquely identifiable) event, the employee who caused the event is recorded, as well as a timestamp and lifecycle information. The latter is provided both in the form of the standard XES lifecycle as well as the internally used lifecycle events.
Of course, the data is fully anonymized. However, the company can map the IDs from the public event log to their own system IDs, down to the event level.

The data is provided in two files:

The Application event log. This event log contains all events with the application as the case ID. Any event related to an offer also refers to an OfferID.
The Offer event log. This event log contains all events related to offers, with these offers as case ID. For each offer, a corresponding application is available.
Please note that there may be multiple offers per application. However, at most one of them should always be accepted.

The XES log files are strictly conforming to the IEEE XES standard and can be loaded in any tool that is compliant with XES.