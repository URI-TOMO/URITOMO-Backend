Analyze the following text to see if it contains ambiguous, high-risk, or culturally loaded expressions that might be misunderstood.

## Context
- Source Language: {{source_lang}}
- Cultural Cards Matched: {{related_cards}}

## Input Text
{{text}}

## Decision Logic
1. Is there a specific Japanese/Korean business phrase used (e.g., "Kentou shimasu")?
2. Is there a high risk of "Yes" meaning "No"?
3. Is it a vague commitment that sounds like a promise in translation?

Output JSON:
{
  "should_explain": boolean,
  "reason": "short string",
  "risk_level": "low" | "medium" | "high"
}
