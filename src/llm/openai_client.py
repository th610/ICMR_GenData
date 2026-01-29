"""
OpenAI API client wrapper with retry logic and JSON parsing.
"""
import os
import json
import time
from typing import Dict, Any, Optional, List
from openai import OpenAI


class OpenAIClient:
    """Wrapper for OpenAI API calls with error handling and JSON parsing."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7, 
                 max_tokens: int = 300, timeout: int = 30, max_retries: int = 1):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Number of retries for JSON parsing failures
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
    
    def call(self, system_prompt: str, user_prompt: str, 
             retry_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Call OpenAI API and parse JSON response.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            retry_message: Message to send if JSON parsing fails (for 1 retry)
        
        Returns:
            Parsed JSON dict
        
        Raises:
            ValueError: If JSON parsing fails after retries
            Exception: If API call fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError as e:
                    # JSON parsing failed
                    if attempt < self.max_retries and retry_message:
                        # Retry with error message
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": retry_message})
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        raise ValueError(f"JSON parsing failed after {attempt + 1} attempts: {e}\nContent: {content}")
            
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    raise Exception(f"API call failed after {attempt + 1} attempts: {e}")
        
        raise ValueError("Max retries exceeded")
    
    def call_batch(self, system_prompt: str, user_prompts: List[str], 
                   retry_message: Optional[str] = None, 
                   verbose: bool = True) -> List[Optional[Dict[str, Any]]]:
        """
        Call API for multiple prompts sequentially.
        
        Args:
            system_prompt: System message (same for all)
            user_prompts: List of user messages
            retry_message: Retry message for JSON failures
            verbose: Print progress
        
        Returns:
            List of parsed JSON dicts (None for failures)
        """
        results = []
        failed_count = 0
        
        for i, user_prompt in enumerate(user_prompts):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(user_prompts)} (failed: {failed_count})")
            
            try:
                result = self.call(system_prompt, user_prompt, retry_message)
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed on item {i + 1}: {e}")
                results.append(None)
                failed_count += 1
        
        if verbose:
            print(f"  Completed: {len(user_prompts) - failed_count}/{len(user_prompts)} successful")
        
        return results
