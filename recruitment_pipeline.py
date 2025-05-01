import os
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import itertools
# from langchain.chat_models import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

@dataclass
class AgentPrompt:
    """Represents a prompt configuration for an agent"""
    name: str
    style: str
    template: str

@dataclass
class AgentResult:
    """Stores the result from an agent"""
    agent_name: str
    output: Any

@dataclass
class PipelineResult:
    """Stores the results from a full pipeline run"""
    sequence: str
    a_result: AgentResult
    b_result: AgentResult
    c_result: AgentResult
    final_rank: Any

class RecruitmentAgent:
    """Base class for all recruitment agents"""
    
    def __init__(self, prompt: AgentPrompt, model_name: str = "claude-3-opus-20240229"):
        self.prompt = prompt
        if "claude" in model_name.lower():
            # self.llm = ChatAnthropic(model=model_name)
            pass
        else:
            self.llm = ChatOpenAI(model=model_name)
    
    def run(self, inputs: Dict[str, Any]) -> Any:
        """Run the agent with the given inputs"""
        prompt_template = PromptTemplate.from_template(self.prompt.template)
        formatted_prompt = prompt_template.format(**inputs)
        result = self.llm([HumanMessage(content=formatted_prompt)])
        return result.content

class KeywordExtractionAgent(RecruitmentAgent):
    """Agent for extracting keywords from resume (Category A)"""
    
    def run(self, resume: str) -> Dict[str, List[str]]:
        """Extract keywords from resume"""
        result = super().run({"resume": resume})
        try:
            # Try to parse as JSON if possible
            return json.loads(result)
        except:
            # If not JSON, return as is
            return {"extracted_text": result}

class CandidateFilteringAgent(RecruitmentAgent):
    """Agent for filtering candidates (Category B)"""
    
    def run(self, keywords: Dict[str, List[str]], job_description: str) -> Dict[str, Any]:
        """Filter candidate based on keywords and job description"""
        result = super().run({
            "keywords": json.dumps(keywords),
            "job_description": job_description
        })
        try:
            # Try to parse as JSON if possible
            return json.loads(result)
        except:
            # If not JSON, return as is
            return {"filtered_result": result}

class CandidateRankingAgent(RecruitmentAgent):
    """Agent for ranking candidates (Category C)"""
    
    def run(self, filtered_profile: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Rank candidate based on filtered profile and job description"""
        result = super().run({
            "filtered_profile": json.dumps(filtered_profile),
            "job_description": job_description
        })
        try:
            # Try to parse as JSON if possible
            return json.loads(result)
        except:
            # If not JSON, return as is
            return {"ranking_result": result}

class AgentFactory:
    """Factory for creating agents with different prompting styles"""
    
    @staticmethod
    def create_keyword_extraction_agents() -> List[KeywordExtractionAgent]:
        """Create two different keyword extraction agents"""
        prompts = [
            AgentPrompt(
                name="A1",
                style="Concise and direct",
                template="""
                Extract the most important skills, qualifications, and experience from the following resume.
                Be concise and list only the key information.
                Format your response as a JSON object with the following structure:
                {
                    "technical_skills": ["skill1", "skill2", ...],
                    "soft_skills": ["skill1", "skill2", ...],
                    "education": ["degree1", "degree2", ...],
                    "experience": ["exp1", "exp2", ...]
                }
                
                Resume:
                {resume}
                """
            ),
            AgentPrompt(
                name="A2",
                style="Detailed and conversational",
                template="""
                I'd like you to carefully analyze this resume and extract comprehensive information about the candidate.
                Please take your time to thoroughly identify all skills, experiences, qualifications, and other relevant information.
                Consider both explicit mentions and implied skills based on their work history.
                
                For technical skills, look for programming languages, tools, platforms, methodologies, etc.
                For soft skills, identify traits like leadership, communication, project management, etc.
                For education, capture degrees, institutions, relevant coursework, certifications, etc.
                For experience, summarize roles, responsibilities, achievements, and duration.
                
                Please structure your output as a JSON object with these categories:
                {
                    "technical_skills": ["skill1", "skill2", ...],
                    "soft_skills": ["skill1", "skill2", ...],
                    "education": ["degree1", "degree2", ...],
                    "experience": ["exp1", "exp2", ...],
                    "additional_keywords": ["keyword1", "keyword2", ...]
                }
                
                Resume to analyze:
                {resume}
                """
            )
        ]
        
        return [KeywordExtractionAgent(prompt) for prompt in prompts]
    
    @staticmethod
    def create_candidate_filtering_agents() -> List[CandidateFilteringAgent]:
        """Create two different candidate filtering agents"""
        prompts = [
            AgentPrompt(
                name="B1",
                style="Analytical and structured",
                template="""
                Analyze the candidate's keywords against the job description.
                Determine if the candidate meets the basic qualifications and requirements.
                
                Keywords from resume:
                {keywords}
                
                Job Description:
                {job_description}
                
                Provide your assessment as a JSON with the following structure:
                {
                    "meets_requirements": true/false,
                    "match_score": 0-100,
                    "strengths": ["strength1", "strength2", ...],
                    "gaps": ["gap1", "gap2", ...],
                    "overall_assessment": "brief statement"
                }
                """
            ),
            AgentPrompt(
                name="B2",
                style="Evaluative and detailed",
                template="""
                You are an experienced technical recruiter evaluating a candidate for a position.
                
                First, carefully review the job description to understand what skills and experience are required.
                Next, examine the candidate's keywords extracted from their resume.
                
                Job Description:
                {job_description}
                
                Candidate's Keywords:
                {keywords}
                
                Your task:
                1. Compare the candidate's profile against each requirement in the job description
                2. Identify specific matches between requirements and the candidate's qualifications
                3. Note any missing required skills or qualifications
                4. Evaluate how well the candidate's experience aligns with the role's responsibilities
                5. Consider both technical and soft skills alignment
                
                Provide a comprehensive evaluation in JSON format:
                {
                    "meets_requirements": true/false,
                    "match_score": 0-100,
                    "requirement_analysis": [
                        {"requirement": "req1", "met": true/false, "evidence": "..."},
                        {"requirement": "req2", "met": true/false, "evidence": "..."},
                        ...
                    ],
                    "key_strengths": ["strength1", "strength2", ...],
                    "key_gaps": ["gap1", "gap2", ...],
                    "suitability_assessment": "detailed paragraph"
                }
                """
            )
        ]
        
        return [CandidateFilteringAgent(prompt) for prompt in prompts]
    
    @staticmethod
    def create_candidate_ranking_agents() -> List[CandidateRankingAgent]:
        """Create two different candidate ranking agents"""
        prompts = [
            AgentPrompt(
                name="C1",
                style="Numerical and objective",
                template="""
                Based on the filtered candidate profile and job description, assign a numerical ranking to the candidate.
                
                Filtered Candidate Profile:
                {filtered_profile}
                
                Job Description:
                {job_description}
                
                Provide a ranking score from 1-10 (10 being the highest) and justify your score.
                Format your response as a JSON:
                {
                    "ranking_score": 1-10,
                    "justification": "brief explanation"
                }
                """
            ),
            AgentPrompt(
                name="C2",
                style="Comprehensive and contextual",
                template="""
                As a senior hiring manager, evaluate this candidate's suitability for the role based on their filtered profile.
                Consider not just technical fit, but also potential culture fit, growth potential, and long-term value.
                
                Filtered Candidate Profile:
                {filtered_profile}
                
                Job Description:
                {job_description}
                
                Provide a detailed multi-dimensional assessment of the candidate with these components:
                
                1. Overall fit score (1-10 scale, 10 being perfect)
                2. Technical capability score (1-10)
                3. Experience relevance score (1-10)
                4. Potential growth/learning curve score (1-10)
                5. Detailed justification for each score
                6. Final recommendation (Reject, Consider, Interview, Strong Recommend)
                
                Return your evaluation as a JSON object:
                {
                    "overall_ranking": 1-10,
                    "technical_score": 1-10,
                    "experience_score": 1-10,
                    "potential_score": 1-10,
                    "justification": {
                        "technical": "explanation",
                        "experience": "explanation",
                        "potential": "explanation"
                    },
                    "recommendation": "one of the four options",
                    "additional_notes": "any other observations"
                }
                """
            )
        ]
        
        return [CandidateRankingAgent(prompt) for prompt in prompts]

class RecruitmentPipeline:
    """Main pipeline for running the recruitment process"""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        self.model_name = model_name
        # Create all agents
        self.a_agents = AgentFactory.create_keyword_extraction_agents()
        self.b_agents = AgentFactory.create_candidate_filtering_agents()
        self.c_agents = AgentFactory.create_candidate_ranking_agents()
        
    def run_all_pipelines(self, resume: str, job_description: str) -> List[PipelineResult]:
        """Run all possible combinations of agents"""
        results = []
        
        # Generate all possible combinations (2x2x2=8)
        for a_agent, b_agent, c_agent in itertools.product(self.a_agents, self.b_agents, self.c_agents):
            sequence = f"{a_agent.prompt.name}->{b_agent.prompt.name}->{c_agent.prompt.name}"
            print(f"Running sequence: {sequence}")
            
            # Stage A: Extract keywords
            a_output = a_agent.run(resume)
            a_result = AgentResult(a_agent.prompt.name, a_output)
            
            # Stage B: Filter candidate
            b_output = b_agent.run(a_output, job_description)
            b_result = AgentResult(b_agent.prompt.name, b_output)
            
            # Stage C: Rank candidate
            c_output = c_agent.run(b_output, job_description)
            c_result = AgentResult(c_agent.prompt.name, c_output)
            
            # Extract the final rank
            if isinstance(c_output, dict):
                final_rank = c_output.get("ranking_score") or c_output.get("overall_ranking")
            else:
                final_rank = None
                
            pipeline_result = PipelineResult(
                sequence=sequence,
                a_result=a_result,
                b_result=b_result,
                c_result=c_result,
                final_rank=final_rank
            )
            
            results.append(pipeline_result)
            
        return results
    
    def evaluate_results(self, results: List[PipelineResult]) -> List[Tuple[str, Any]]:
        """Evaluate and sort the results of all pipelines"""
        # Sort by final rank (higher is better)
        sorted_results = sorted(
            [(r.sequence, r.final_rank) for r in results],
            key=lambda x: x[1] if x[1] is not None else 0,
            reverse=True
        )
        
        return sorted_results

def main(resume_path: str, job_description_path: str):
    """Main function to run the recruitment pipeline"""
    # Load resume and job description
    with open(resume_path, 'r', encoding='utf-8') as f:
        resume = f.read()
        
    with open(job_description_path, 'r', encoding='utf-8') as f:
        job_description = f.read()
    
    # Initialize pipeline
    pipeline = RecruitmentPipeline()
    
    # Run all pipelines
    results = pipeline.run_all_pipelines(resume, job_description)
    
    # Evaluate results
    sorted_results = pipeline.evaluate_results(results)
    
    # Print sorted results
    print("\nSorted Pipeline Results (by ranking score):")
    for sequence, rank in sorted_results:
        print(f"{sequence}: {rank}")
    
    # Save detailed results to file
    with open('pipeline_results.json', 'w') as f:
        json.dump(
            [{
                "sequence": r.sequence,
                "a_output": r.a_result.output,
                "b_output": r.b_result.output,
                "c_output": r.c_result.output,
                "final_rank": r.final_rank
            } for r in results],
            f,
            indent=2
        )
    
    print(f"\nDetailed results saved to pipeline_results.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the recruitment pipeline")
    parser.add_argument("resume", help="Path to the resume file")
    parser.add_argument("job_description", help="Path to the job description file")
    
    args = parser.parse_args()
    
    main(args.resume, args.job_description) 