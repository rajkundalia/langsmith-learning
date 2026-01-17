# LangSmith Learning Project - UI-Focused Evaluation Framework

A hands-on learning project that teaches LangSmith through its most powerful feature: the visual UI for tracing and evaluating LLM applications.

**Core Philosophy:** This project emphasizes exploring the LangSmith web interface rather than writing complex code. After running the script, you'll spend most of your time in the dashboard discovering insights about your LLM's behavior.

## What You'll Learn

- ✅ How to trace LLM calls and inspect execution details
- ✅ How to create evaluation datasets programmatically
- ✅ How to write custom evaluators for different quality metrics
- ✅ How to run systematic experiments comparing prompts and configurations
- ✅ How to use the LangSmith UI to analyze, debug, and improve LLM applications
- ✅ The iterative workflow of LLM development: experiment → evaluate → feedback → improve

## Why LangSmith?

### The Problems It Solves

**Black-Box Debugging**  
LLMs are opaque. When your chatbot gives a wrong answer, where do you even start? LangSmith lets you see inside every LLM call: the exact prompt sent, the response received, token usage, latency, and more. No more guessing what went wrong.

**Systematic Evaluation**  
Manual testing doesn't scale. You might test 3-5 examples by hand, but what about the 100 edge cases? LangSmith runs evaluations across dozens or hundreds of examples automatically, giving you quantified metrics instead of gut feelings.

**Objective Comparison**  
"This prompt seems better" is subjective. LangSmith lets you A/B test prompts with concrete metrics: Prompt A scores 0.85 on relevance vs Prompt B's 0.72. Make decisions based on data, not intuition.

**Production Visibility**  
Once deployed, how do you know if your LLM app is working well? LangSmith monitors real usage, catches issues early, and helps you understand what users are actually asking and how well your system responds.

### When Should You Use LangSmith?

- **Building any LLM application** - Even prototypes benefit from tracing. Start early, avoid painful debugging later.
- **Trying to improve prompt quality** - Systematic evaluation beats trial-and-error every time.
- **Comparing different models or configurations** - Test GPT-4 vs Claude vs local models with real metrics.
- **Debugging unexpected LLM behavior** - See exactly what the model received and returned.
- **Monitoring production applications** - Track performance, catch regressions, identify problem patterns.

### LangSmith vs Alternatives

| Feature | Print Debugging | Manual Testing | Custom Logging | LangSmith |
|---------|----------------|----------------|----------------|-----------|
| **Visual traces** | ❌ | ❌ | ❌ | ✅ |
| **Persistent history** | ❌ | ❌ | ⚠️ (DIY) | ✅ |
| **Shareable with team** | ❌ | ❌ | ⚠️ (DIY) | ✅ |
| **Reproducible tests** | ❌ | ❌ | ❌ | ✅ |
| **Scales to 100s of examples** | ❌ | ❌ | ⚠️ (hard) | ✅ |
| **Quantified results** | ❌ | ⚠️ (manual) | ⚠️ (DIY) | ✅ |
| **Built-in comparison tools** | ❌ | ❌ | ❌ | ✅ |
| **No maintenance burden** | ✅ | ✅ | ❌ | ✅ |

**Bottom line:** LangSmith gives you professional observability tools without building your own infrastructure.

## Prerequisites

Before starting, make sure you have:

- **LangSmith account** - Sign up at [smith.langchain.com](https://smith.langchain.com) (free tier available)
- **Ollama** with llama3 model:
  ```bash
  # Install from: https://ollama.ai
  # Then pull the model:
  ollama pull llama3
  ```
- **Python 3.10+**
- **Basic understanding** of LLMs and prompt engineering

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd langsmith-learning

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Edit .env and add your LangSmith API key
# Get it from: https://smith.langchain.com/settings
```

### CRITICAL: Environment Configuration

Your `.env` file **must** contain these three variables:

```bash
LANGSMITH_API_KEY=lsv2_pt_your_actual_key_here
LANGSMITH_PROJECT=langsmith-learning
LANGSMITH_TRACING_V2=true  # Enables automatic tracing
```

**Without `LANGSMITH_TRACING_V2=true`, you won't see any traces in the LangSmith UI!** This is the most common mistake when starting with LangSmith.

## Quick Start

```bash
# Make sure Ollama is running
ollama serve

# Run the learning project
python main.py
```

### What Happens During Execution

The script runs through six phases:

1. **Setup** - Connects to LangSmith and Ollama
2. **Dataset creation** - Uploads 6 QA pairs
3. **Initial trace generation** - Creates sample traces for exploration
4. **Interactive pause** - You explore the UI (take your time!)
5. **Prompt comparison** - Tests 3 prompts × 6 examples × 4 evaluators
6. **Temperature comparison** - Tests best prompt with 3 temperature settings
7. **Feedback loop demo** - Shows how to iterate with feedback

## What You'll See in the LangSmith UI

This section explains each tab and feature you'll encounter.

### Projects Tab

Your "LLM call history" - every traced run appears here.

**Key features:**
- Filter by date, status, tags, or search
- Click any run to see detailed trace
- Group by experiment to compare runs
- Export data for further analysis

**What to look for:**
- Which runs succeeded vs failed
- Latency patterns across runs
- Token usage and estimated costs

### Trace View (Most Important!)

The detailed view of a single LLM call showing exactly what happened.

**Trace structure:**
```
qa_chain (root)
├─ Input: "What is photosynthesis?"
├─ LLM Call (llama3)
│  ├─ Prompt: "Answer the following question..."
│  ├─ Response: "Photosynthesis is the process..."
│  ├─ Latency: 1.2s
│  └─ Tokens: 45 in, 78 out
└─ Output: "Photosynthesis is the process..."
```

**Each node shows:**
- **Input and output** - What went in and came out
- **Latency** - How long each step took
- **Token usage** - Input/output tokens, cost estimation
- **Metadata** - Custom info attached to runs
- **Nested calls** - For chains, see parent-child relationships

**How to navigate:**
- Click on any node to expand details
- Use the tree view to understand execution flow
- Check the "Feedback" section to see ratings

**Pro tip:** Click through several traces of the same question answered by different prompts to see how prompt design affects behavior.

### Datasets Tab

Your uploaded QA pairs and test cases.

**Features:**
- Edit examples directly in the UI
- Add new examples without touching code
- Version history tracks changes over time
- Export datasets for sharing

**Use cases:**
- Refine your test cases based on findings
- Add edge cases you discover in production
- Build regression test suites iteratively

### Experiments & Comparisons

Side-by-side metrics for different runs - the heart of systematic evaluation.

**What you'll see:**
- All three prompts tested on the same questions
- Metrics table showing scores for each evaluator
- Statistical summaries (mean, median, std dev)
- Visual charts comparing performance

**Example view:**
```
Experiment Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              Relevance  Concise  Correct  Tone
Prompt A        0.85      0.78     0.71    0.82
Prompt B        0.92      0.65     0.79    0.88
Prompt C        0.81      0.95     0.68    0.75
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Winner:       Prompt B  Prompt C Prompt B Prompt B
```

**How to use:**
- Identify which prompt wins on which metrics
- Spot tradeoffs (Prompt B: high relevance, low conciseness)
- Export results as CSV for reports
- Make data-driven decisions about which prompt to use

### Feedback Tab

Human-in-the-loop evaluation - your thumbs up/down on individual traces.

**How to give feedback:**
1. Find a trace you find interesting
2. Click the 👍 or 👎 button
3. Add a comment: "Too verbose" or "Perfect explanation"

**Why it matters:**
- Feedback aggregates across runs
- Filter traces by feedback to find patterns
- Use feedback to build datasets of good/bad examples
- Track whether changes improve thumbs-up ratio

**Pattern example:**
```
All 👎 traces involve ambiguous questions
→ Insight: Need better handling of ambiguity
→ Action: Add clarifying questions to prompt
```

## Understanding the Evaluators

Each evaluator measures a different aspect of quality. Here's what they do:

### Relevance Evaluator

**What it measures:** Does the output actually answer the question?

**Scoring logic:**
- Checks for keyword overlap between question and answer
- Verifies output has meaningful content (not just "I don't know")
- Awards partial credit for tangentially related answers

**Score interpretation:**
- **0.8-1.0** (High): Output directly addresses the question
- **0.5-0.7** (Medium): Output is related but not directly answering
- **<0.5** (Low): Output is off-topic or too generic

**Example:**
- Q: "What is democracy?"
- A: "Democracy involves voting and representation" → **High score** (0.8)
- A: "Government systems are complex" → **Low score** (0.3)

### Conciseness Evaluator

**What it measures:** Is the answer appropriately brief?

**Scoring logic:**
- Counts words in the output
- Penalizes both too-short and too-long answers
- Sweet spot: 20-100 words

**Score interpretation:**
- **1.0** (Ideal): 20-100 words - good detail-to-length ratio
- **0.7** (Acceptable): 10-20 words OR 100-200 words
- **0.3-0.4** (Poor): <10 words (too brief) OR >200 words (rambling)

**Example:**
- Q: "What is 2+2?"
- A: "4" → **Low score** (0.3 - too brief, no explanation)
- A: [150-word essay on addition] → **Low score** (0.4 - way too verbose)
- A: "The answer is 4. In mathematics, addition combines numbers." → **High score** (1.0)

### Factual Correctness Evaluator

**What it measures:** Accuracy compared to the expected answer.

**Scoring logic:**
- Compares output to the `expected_output` in your dataset
- Uses keyword matching to find key facts
- Awards full points for hitting most key facts

**Score interpretation:**
- **0.8-1.0** (High): Contains 80%+ of key facts from expected output
- **0.5-0.7** (Medium): Contains 50-80% of key facts
- **<0.5** (Low): Missing critical facts or contains errors

**Example:**
- Q: "What is the capital of France?"
- Expected: "Paris"
- A: "The capital of France is Paris" → **High score** (1.0)
- A: "The capital is Lyon" → **Low score** (0.3 - factually wrong)

**Note:** This is a simple implementation. Production systems might use semantic similarity (embeddings) or LLM-as-a-judge for better accuracy.

### Tone/Professionalism Evaluator

**What it measures:** Appropriate, helpful communication style.

**Scoring logic:**
- Checks for condescending language ("obviously", "just do this")
- Penalizes overly casual ("lol", "gonna") or stiff language
- Rewards helpful phrasing ("let me explain", "here's how")

**Score interpretation:**
- **0.8-1.0** (High): Clear, respectful, neither too casual nor too formal
- **0.6-0.7** (Medium): Minor tone issues
- **<0.6** (Low): Condescending, inappropriate formality, or unhelpful

**Example:**
- "Obviously, anyone knows that democracy means voting" → **Low score** (0.5 - condescending)
- "Let me explain: democracy is a system where citizens vote" → **High score** (0.9 - helpful tone)

**Why this matters:** Shows you can evaluate subjective criteria, not just factual correctness. In production, tone might be as important as accuracy for user satisfaction.

## Exploring Results - Step by Step

Here's exactly how to use this project for maximum learning:

### 1. Run the Script

```bash
python main.py
```

Watch the console output for status updates and URLs.

### 2. Explore Initial Traces (First Pause)

When you see "EXPLORE THE UI NOW", do this:

1. **Open the printed URL** in your browser
2. **Click through a trace** to see the execution flow
3. **Examine the structure:**
   - Root node (qa_chain): Overall function
   - LLM Call node: The actual model invocation
   - Expand each to see input/output
4. **Check metrics:**
   - Latency: How long did it take?
   - Tokens: How many input/output tokens?
   - Cost: Estimated cost (if using paid models)
5. **Look at metadata:** Any custom tags or info attached

**Don't rush this step!** Understanding traces is fundamental to everything else.

### 3. Continue to Experiments

Press Enter when ready. The script will:
- Run prompt comparison (~2-3 minutes)
- Test 3 different prompts on 6 questions
- Apply 4 evaluators to each run

You'll see progress in the console:
```
⚙️  Testing Prompt A (Neutral)...
   ✓ Prompt A complete
     Avg scores: Rel=0.85 | Conc=0.78 | Corr=0.71 | Tone=0.82
```

### 4. Compare Prompts Side-by-Side

Open the experiments comparison URL and explore:

**Look for:**
- Which prompt scored highest on each evaluator
- Tradeoffs: Prompt B might win on relevance but lose on conciseness
- Consistency: Does one prompt have less variation in scores?

**Try this:**
1. Click "Compare" view to see all three prompts side-by-side
2. Filter by a specific question to see how each prompt answered it
3. Look at the detailed metrics table
4. Export results as CSV if you want to analyze further

### 5. Examine Individual Runs

Don't just look at averages - click into specific traces:

1. Find a trace where evaluators disagreed (high relevance, low correctness)
2. Click through to see the actual LLM response
3. Read the evaluator comments to understand why it scored that way
4. Compare this trace to others for the same question

**Insight example:** You might discover that Prompt C gives concise answers but misses important nuances that Prompt B captures.

### 6. Give Feedback (Optional but Recommended)

Practice the feedback loop:

1. Find a trace you find particularly good or bad
2. Click the 👍 or 👎 button
3. Add a comment:
   - "This answer was too verbose for a simple question"
   - "Perfect balance of detail and clarity"
   - "Missing the key point about X"
4. Check the Feedback tab to see your ratings aggregated

This feedback shows up in your project and can inform the next iteration.

### 7. Review Temperature Comparison

After the temperature experiments complete:

**Notice:**
- Lower temperature (0.3): More consistent, factual, but potentially rigid
- Higher temperature (1.0): More varied, creative, but less predictable

**Compare runs:**
- Run the same question through different temperatures
- See how outputs change in style and content
- Decide which temperature suits your use case

### 8. Plan Your Next Iteration

Based on what you learned:
- Which prompt performed best overall?
- What tradeoffs matter for your use case?
- Which temperature setting should you use?
- What examples should you add to your dataset?

## Common Pitfalls & Solutions

### "I don't see any traces in LangSmith"

This is the #1 issue for beginners. Check these in order:

✅ **MOST COMMON:** Verify `LANGSMITH_TRACING_V2=true` is in your `.env` file  
✅ Confirm `LANGSMITH_API_KEY` in `.env` is correct (copy from https://smith.langchain.com/settings)  
✅ Check `LANGSMITH_PROJECT` name matches what you see in the UI  
✅ Look for connection errors in console output  
✅ Refresh the LangSmith dashboard  
✅ Verify you're looking at the right project (dropdown in top-left of UI)  
✅ Restart your script after modifying `.env` (environment vars load at startup)

**Debug command:**
```bash
# Check if environment variables are loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('LANGSMITH_TRACING_V2'))"
# Should print: true
```

### "Evaluators aren't running"

✅ Check evaluators are passed to `evaluate()` function  
✅ Verify evaluator function signatures: `(run: Run, example: Example) -> dict`  
✅ Look for errors in console output  
✅ Ensure return dict has 'key', 'score', and 'comment' fields  
✅ Check if evaluator is imported correctly in `main.py`

### "Can't find my dataset"

✅ Check dataset name matches exactly (case-sensitive)  
✅ Look in the Datasets tab in LangSmith UI  
✅ Verify dataset was created successfully (check console output)  
✅ Try the dataset URL printed by the script  
✅ Make sure `datasets/qa_pairs.json` exists and is valid JSON

### "Ollama errors or timeouts"

✅ Ensure Ollama is running: `ollama serve`  
✅ Verify llama3 model is pulled: `ollama pull llama3`  
✅ Check Ollama is accessible: `curl http://localhost:11434`  
✅ Try `ollama list` to see installed models  
✅ If still failing, check Ollama logs for errors

### "Evaluator scores all seem the same"

This might be legitimate:

- Your evaluators might need refinement (tweak scoring thresholds)
- Your prompts might actually be performing similarly
- Check individual traces to see the actual outputs
- Look at the comments field for more nuance

**Fix:** Modify evaluators in `evaluators.py` to be more discriminating.

## Try These Extensions

Once you've completed the basic project, try these to deepen your learning:

### Extension 1: Add Your Own Evaluator

Create a "clarity" evaluator that checks if responses avoid jargon:

```python
def clarity_evaluator(run: Run, example: Example) -> dict:
    """Penalize technical jargon, reward simple explanations."""
    output = run.outputs.get("output", "").lower()
    
    # Check for complex words or jargon
    jargon_words = ["synergistic", "paradigm", "utilize", "leverage"]
    jargon_count = sum(1 for word in jargon_words if word in output)
    
    score = max(0, 1.0 - (jargon_count * 0.2))
    
    return {
        "key": "clarity",
        "score": score,
        "comment": f"Found {jargon_count} jargon words"
    }
```

Add it to `evaluators.py` and import it in `main.py`.

### Extension 2: Test Different Models

Compare llama3 vs llama3:70b (if you have resources):

```python
# After pulling: ollama pull llama3:70b
llm_small = ChatOllama(model="llama3")
llm_large = ChatOllama(model="llama3:70b")

# Run same experiments with both
# Question: Does the larger model score better? Is it worth the cost?
```

### Extension 3: Real-World Dataset

Replace the toy QA pairs with real questions from your domain:

```python
# For a customer support bot:
qa_pairs = [
    {
        "input": "How do I reset my password?",
        "expected_output": "Click 'Forgot Password' on the login page..."
    },
    {
        "input": "What's your refund policy?",
        "expected_output": "We offer full refunds within 30 days..."
    },
    # Add 10-20 real questions from your support tickets
]
```

Save to `datasets/my_custom_qa.json` and modify `main.py` to load it.

### Extension 4: Production Monitoring Pattern

Wrap a production function with tracing:

```python
from langsmith import traceable

@traceable(name="customer_support_bot")
def handle_support_ticket(question: str) -> str:
    # Your production logic
    response = llm.invoke(f"Help with: {question}")
    return response

# In production, every call gets traced automatically
# Add online evaluators that run on each production call
# Set up alerts for low scores in the LangSmith UI
```

### Extension 5: Dataset from Traces (Advanced)

After running experiments, turn failures into test cases:

1. **Find interesting traces** in the UI where evaluators disagreed
2. **Export them** as a new dataset
3. **Use this dataset** to test prompt improvements
4. **Create a regression test suite** from real failures

This creates a virtuous cycle: production issues → test cases → prevent regressions.

## Next Steps

Now that you understand LangSmith basics:

### Integrate into Your Projects

- Add `@traceable` decorators to your LLM calls
- Start with just tracing, add evaluators later
- Even without evaluators, traces are valuable for debugging

### Build Domain-Specific Evaluators

Think about what "quality" means for your use case:
- Customer support: Empathy, clarity, action items
- Code generation: Correctness, efficiency, readability
- Content writing: Engagement, accuracy, brand voice

### Create a Regression Test Suite

Start small:
1. Collect 10-20 examples from real usage
2. Define what "good" looks like for each
3. Run evaluations on every prompt change
4. Grow the suite as you find new edge cases

### Set Up Production Monitoring

Use LangSmith in production:
- Enable tracing for all production calls
- Set up online evaluators (run automatically)
- Monitor feedback trends
- Set up alerts for quality degradation

### Explore Advanced Features

- **Comparison views** - A/B test prompts with real metrics
- **Online evaluators** - Run automatically on production traces
- **Feedback-driven datasets** - Turn user feedback into test cases
- **Team collaboration** - Share traces and experiments with teammates

## Resources

- **[LangSmith Documentation](https://docs.smith.langchain.com)** - Official docs, comprehensive reference
- **[LangSmith Cookbook](https://github.com/langchain-ai/langsmith-cookbook)** - Advanced examples and patterns
- **[LangChain Documentation](https://python.langchain.com)** - For building chains and agents
- **[Prompt Engineering Guide](https://www.promptingguide.ai)** - Improve your prompts
- **[Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)** - Best practices from Claude creators

## FAQ

**Q: Do I need to use LangChain to use LangSmith?**  
A: No! LangSmith works with any LLM framework (OpenAI SDK, Anthropic SDK, etc.). LangChain just makes it easier to get started with automatic tracing.

**Q: Can I use LangSmith with OpenAI/Anthropic/other providers?**  
A: Yes! This project uses Ollama for simplicity and cost, but LangSmith works with all major providers:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
# Tracing works automatically
```

**Q: Is LangSmith only for evaluation, or can I use it in production?**  
A: Both! Many teams use LangSmith to monitor production traffic. It's not just a dev tool.

**Q: How much does LangSmith cost?**  
A: There's a generous free tier perfect for learning and small projects. Check [smith.langchain.com/pricing](https://smith.langchain.com/pricing) for current details.

**Q: Can I share my LangSmith project with teammates?**  
A: Yes! Invite team members in the UI, share trace URLs, collaborate on datasets and experiments.

**Q: Why use evaluators instead of just looking at outputs manually?**  
A: Manual review doesn't scale. Evaluators let you test 100 examples in minutes instead of hours, track metrics over time, and make objective comparisons.

**Q: Can I use LangSmith with agents and complex chains?**  
A: Absolutely! LangSmith shines with complex applications. Traces show the full execution tree, making it easier to debug multi-step workflows.

**Q: What if my LLM calls are asynchronous?**  
A: LangSmith fully supports async. The `@traceable` decorator works with async functions, and traces still capture the full execution.

## Contributing

Found a bug? Have an improvement? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - feel free to use this for learning or as a base for your own projects.

---

**Remember:** The goal isn't just to run the code, but to develop the mindset of systematic LLM evaluation. Spend time in the LangSmith UI. Explore traces. Compare experiments. Give feedback. Iterate. That's where the real learning happens! 🚀