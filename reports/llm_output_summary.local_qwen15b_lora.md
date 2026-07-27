# Stage 5 — LLM Output Summary

- Language model: `Qwen/Qwen2.5-1.5B-Instruct` + adapter `adapters/stage5-qwen25-15b-json-lora`
- Prompting template: system instruction forbidding invented attributes, explicit JSON schema description, candidate product list with metadata only
- Number of generated answers: 20
- JSON validity rate (first attempt): 10.00%
- Schema validity rate (after retries): 15.00%
- Number of repaired outputs (needed a retry or fallback): 2

## Example final answers

```json
{
  "query_id": "q001",
  "interpreted_need": {
    "category": "unknown",
    "use_case": "unknown",
    "positive_preferences": [],
    "negative_constraints": [],
    "visual_preferences": [],
    "uncertain_fields": [
      "could not parse model output"
    ]
  },
  "product_judgements": [
    {
      "product_id": "B086B5MRFB|amazon.in",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    },
    {
      "product_id": "B07B4MRLT8|amazon.com",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    },
    {
      "product_id": "B07QJXW4JR|amazon.com",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    }
  ],
  "decision": "ask_clarification",
  "customer_response": "We found some candidate products for your query, but could not confidently structure a recommendation. Please refine your query or check the top results directly."
}
```

```json
{
  "query_id": "q002",
  "interpreted_need": {
    "category": "unknown",
    "use_case": "unknown",
    "positive_preferences": [],
    "negative_constraints": [],
    "visual_preferences": [],
    "uncertain_fields": [
      "could not parse model output"
    ]
  },
  "product_judgements": [
    {
      "product_id": "B07DBF3VJY|amazon.com",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    },
    {
      "product_id": "B07K7NNS6N|amazon.co.uk",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    },
    {
      "product_id": "B07J1YW3YT|amazon.com",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    }
  ],
  "decision": "ask_clarification",
  "customer_response": "We found some candidate products for your query, but could not confidently structure a recommendation. Please refine your query or check the top results directly."
}
```

```json
{
  "query_id": "q003",
  "interpreted_need": {
    "category": "unknown",
    "use_case": "unknown",
    "positive_preferences": [],
    "negative_constraints": [],
    "visual_preferences": [],
    "uncertain_fields": [
      "could not parse model output"
    ]
  },
  "product_judgements": [
    {
      "product_id": "B07SSHYD2Z|amazon.co.uk",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    },
    {
      "product_id": "B081HX9SRM|amazon.co.uk",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    },
    {
      "product_id": "B081HWKWN6|amazon.co.uk",
      "role": "substitute",
      "evidence": [],
      "constraint_violations": [],
      "reason": "Automated fallback: language model output could not be validated."
    }
  ],
  "decision": "ask_clarification",
  "customer_response": "We found some candidate products for your query, but could not confidently structure a recommendation. Please refine your query or check the top results directly."
}
```

## Limitations / failure case

Query `q001` fell back to the deterministic fallback answer after 2 retries; last validation error: invalid JSON: Expecting ',' delimiter: line 1 column 585 (char 584)
