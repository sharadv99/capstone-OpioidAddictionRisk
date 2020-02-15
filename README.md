# Opioid Misuse Risk Tool

--Data Science project at UC Berkeley--
--In collaboration with Cameron Kennedy, Aditi Khullar, Rachel Kramer--

<u>Description</u>

The Opioid Misuse Risk Tool is a custom web-app that uses machine learning to impact the opioid crisis, specifically by enabling physicians to make more informed decisions about prescribing opioids. Patients submit responses to ~25 demographic and health related questions, which triggers the data pipeline to output a report detailing their personalized probability of opioid misuse (risk score), their percentile of misuse among all patients, and the impact that each of their responses has on their risk score. A calibrated XG Boost model produces the risk score, with Shapley values generating the contributions of individualized risk factors.

Our minimum viable product:

- uncovers hidden insights on each patient through sophisticated, predictive modeling
- showcases transparent, actionable results for physician's to utilize when interacting with each patient
- presents each patient's misuse risk likelihood and ranks the distinct attributes driving their results
- processes a user-friendly, concise assessment and outputs results seconds after submission

For a comprehensive discussion on the problem motivating the solution (ie. Opioid Crisis), the tool's technical design/workflow, and instructions for using the tool, please visit our [website](https://opioidmisuserisk.github.io/)

You can access the tool directly from our website, or by clicking [here](https://opioidrisk.herokuapp.com/polls)
