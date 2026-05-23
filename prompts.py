TRAJECTORY_REGENERATION_PROMPT = """You are an AI assistant that generates **FAILURE TRAJECTORY DATA** for fault attribution training in Multi-Agent Systems (MAS).

## GENERATION MODE: Trace Regeneration
Your task is to generate a **COMPLETELY DIFFERENT** failure trajectory for the **EXACT SAME** task.

## KEY OBJECTIVES:
1. **KEEP the same question** - Use the EXACT same task/question
2. **KEEP the same ground truth** - The correct answer remains unchanged
3. **GENERATE a different trajectory** - Create a NEW conversation with DIFFERENT reasoning path and DIFFERENT mistakes
4. **USE the same agents** - Keep the same team of agents

## CRITICAL: PRESERVE THESE EXACTLY
- **Question**: {question}
- **Ground Truth Answer**: {ground_truth}

## Reference Trajectory (for style reference only, generate DIFFERENT content):
{sample_text}

## Agent Team (use the same agents):
{agents_description}

{length_guidance}

## Original Mistake Information (generate a DIFFERENT mistake):
- **Original Mistake Agent**: {mistake_agent}
- **Original Mistake Step**: {mistake_step}  
- **Original Mistake Reason**: {mistake_reason}

## YOUR TASK:
Generate a **NEW** multi-agent conversation that:
1. Solves the **EXACT SAME** question (copy it exactly)
2. Arrives at the **WRONG** answer (not the ground truth)
3. Has a **DIFFERENT** reasoning path than the reference
4. Contains a **DIFFERENT** type of mistake or the same type at a different step
5. Uses the **SAME team of agents** with similar interaction patterns

## CRITICAL REQUIREMENTS:
1. **SAME QUESTION** - The question field MUST be identical to the original
2. **SAME GROUND TRUTH** - Copy the ground_truth exactly  
3. **DIFFERENT TRAJECTORY** - The conversation MUST be different from the reference
4. **DIFFERENT MISTAKE** - The error should be different (different agent, different step, or different type)
5. **THE TASK MUST FAIL** - Agents do NOT reach the correct answer
6. **CLEAR ATTRIBUTION** - Identify which agent caused the failure and when
7. **DIFFERENT MISTAKE POSITION** - The mistake MUST occur at a DIFFERENT step than the original (original was step {mistake_step}). Do NOT place the mistake at the same step number as the reference.

## Variation Strategies (choose one or more):
- Different agent makes the first mistake
- Same agent makes a mistake at a different step
- Different type of error (e.g., calculation error vs. misunderstanding)
- Different reasoning approach that leads to a different failure point
- Different intermediate steps before the mistake

## Output Format (JSON):
```json
{{
    "question": "{question}",
    "ground_truth": "{ground_truth}",
    "history": [
        {{"role": "...", "name": "...", "content": "..."}}
    ],
    "mistake_agent": "Name of agent that made the mistake (can be different from original)",
    "mistake_step": "Step number (0-indexed) where mistake occurred",
    "mistake_reason": "Detailed explanation of what went wrong (should be different from original)"
}}
```

Generate a different failure trajectory for the same task:"""


ALL_AT_ONCE_RANKING_PROMPT = """You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem. The problem is: {problem}

Identify the top 3 steps where mistakes are most likely to have occurred. You do NOT need to identify which agent made the mistake, only the step number.

Here's the conversation:
{chat_content}

Instructions:
1. For each of the top 3 suspected mistakes, provide:
Step Number (the step where the mistake first occurred)
Short Reason (1-2 sentences explaining why this step might be a mistake)
2. Always provide exactly 3 steps, even if some mistakes are not obvious. Do not use 'None'.
3. Ensure that all step numbers are valid indices within the conversation (0 to N-1, where N is the number of steps).
4. Please answer in the following format:

Rank 1 Step: (Step number). Reason: (Short reason)
Rank 2 Step: (Step number). Reason: (Short reason)
Rank 3 Step: (Step number). Reason: (Short reason)

Correct Example (follow this strictly):
Rank 1 Step: 3. Reason: This step introduces an incorrect assumption that affects later reasoning.
Rank 2 Step: 7. Reason: The calculation is based on incomplete information.
Rank 3 Step: 12. Reason: The conclusion is drawn without verifying intermediate results."""