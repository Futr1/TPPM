# Privacy, Ethics, and Safety

## Sensitive Data

Psychological profiles constitute highly sensitive personal data. Any real-world deployment must:

- Obtain **explicit, informed user consent** before collecting or maintaining profiles
- Apply **data minimization** — retain only necessary psychological information
- Use **encryption at rest and in transit** for stored PPMUs
- Provide **user data export and deletion** mechanisms

## Clinical Disclaimer

TPPM is a **supportive memory module**, not a replacement for professional psychological counseling or clinical diagnosis.

- The system does not perform clinical assessment, diagnosis, or treatment
- In high-risk scenarios (self-harm, harm to others, severe crises), the system should **escalate to human intervention**

## Deployment Requirements

- Deploy only after safety evaluation in the specific deployment context
- Maintain human-in-the-loop oversight for high-stakes interactions
- Audit for biases in psychological profile extraction across demographic groups
- Consider cultural differences in psychological expression and interpretation

## API Key Security

- Never commit API keys to version control
- Use environment variables (`DEEPSEEK_API_KEY`)
- Review `.gitignore` excludes before committing
