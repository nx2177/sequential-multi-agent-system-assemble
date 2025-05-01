# sequential-multi-agent-system-assemble

# Multi-Agent Recruitment Pipeline

A system that processes resumes through a multi-agent recruitment pipeline to rank candidates for a job position.

## Overview

This system uses LangChain and LLM APIs (Claude or OpenAI) to process resumes and job descriptions through a pipeline of agents:

1. **Keyword Extraction Agents (A)**: Extract important skills and qualifications from resumes
2. **Candidate Filtering Agents (B)**: Compare candidate keywords to job requirements
3. **Candidate Ranking Agents (C)**: Rank candidates based on their filtered profiles

Each stage has two different agents with distinct prompting styles. The system runs all possible combinations (2×2×2=8) and evaluates which sequence produces the best ranking.

## Requirements

- Python 3.8+
- LangChain
- Claude API key (preferred) or OpenAI API key
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your API keys:
   ```
   # For Claude (Anthropic)
   export ANTHROPIC_API_KEY=your_api_key_here
   
   # For OpenAI
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the pipeline with a resume and job description:

```bash
python recruitment_pipeline.py example_resume.txt example_job_description.txt
```

The script will:
1. Process the resume through all 8 possible agent sequences
2. Evaluate and rank the results
3. Output the rankings to the console
4. Save detailed results to `pipeline_results.json`

## How It Works

1. **Keyword Extraction (Stage A)**:
   - A1: Concise and direct prompt
   - A2: Detailed and conversational prompt

2. **Candidate Filtering (Stage B)**:
   - B1: Analytical and structured prompt
   - B2: Evaluative and detailed prompt

3. **Candidate Ranking (Stage C)**:
   - C1: Numerical and objective prompt
   - C2: Comprehensive and contextual prompt

The system runs all 8 combinations (A1→B1→C1, A1→B1→C2, ..., A2→B2→C2) and evaluates which sequence produces the most effective ranking.

## Customization

You can modify the agent prompts in the `AgentFactory` class to experiment with different prompting strategies.

## Output Format

The `pipeline_results.json` file contains detailed results for each agent sequence, including:
- The full sequence (e.g., "A1->B1->C1")
- Outputs from each stage
- Final ranking score 
