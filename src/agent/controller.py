"""
Controller - 제어 정책

Role: Detector 결과를 받아 최종 응답 결정

Policy Levels:
    - Soft Intervention: 일반 위반 → 최선 후보 선택/가벼운 수정
    - Hard Intervention: 고위험 위반 → 강제 개입 (안전 템플릿/재생성)
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from src.llm.openai_client import OpenAIClient


class InterventionLevel(Enum):
    """개입 레벨"""
    NONE = "none"           # 위반 없음, 그대로 선택
    SOFT = "soft"           # 가벼운 수정 또는 재순위
    HARD = "hard"           # 강제 개입 필요


class InterventionPolicy:
    """정책 규칙 정의"""
    
    # Hard intervention이 필요한 위반들
    CRITICAL_VIOLATIONS = {"v4", "v5"}  # Safety violations
    
    # Soft intervention threshold
    SOFT_THRESHOLD = {"v1": 0.7, "v2": 0.8, "v3": 0.7}
    
    @classmethod
    def determine_level(cls, detection_result: Dict[str, Any]) -> InterventionLevel:
        """
        위반 감지 결과로부터 개입 레벨 결정
        
        Args:
            detection_result: ViolationDetector 출력
        
        Returns:
            InterventionLevel
        """
        top_violation = detection_result["top_violation"]
        confidence = detection_result["confidence"]
        severity = detection_result["severity"]
        
        # Normal - 개입 불필요
        if top_violation == "normal":
            return InterventionLevel.NONE
        
        # Critical violations - Hard intervention
        if top_violation in cls.CRITICAL_VIOLATIONS:
            return InterventionLevel.HARD
        
        # Other violations - Soft intervention
        return InterventionLevel.SOFT


class Controller:
    """응답 선택/수정 정책 실행기"""
    
    def __init__(self, 
                 llm_client: Optional[OpenAIClient] = None,
                 enable_modification: bool = True):
        """
        Args:
            llm_client: 수정/재생성용 LLM
            enable_modification: 수정 기능 활성화 여부
        """
        self.llm = llm_client
        self.enable_modification = enable_modification
        self.policy = InterventionPolicy
    
    def select_response(self, 
                       candidates: List[Dict[str, Any]], 
                       detection_results: List[Dict[str, Any]],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        후보들과 위반 감지 결과로부터 최종 응답 결정
        
        Args:
            candidates: CandidateGenerator 출력
            detection_results: ViolationDetector 출력 (각 후보별)
            context: ContextBuilder 출력
        
        Returns:
            {
                "final_response": str,
                "selected_candidate_id": int,
                "intervention_level": str,
                "intervention_details": {...},
                "all_candidates": [...],     # 디버깅용
                "all_detections": [...]      # 디버깅용
            }
        """
        # 1. 후보별 개입 레벨 결정
        candidate_scores = []
        for i, (cand, detect) in enumerate(zip(candidates, detection_results)):
            level = self.policy.determine_level(detect)
            candidate_scores.append({
                "candidate": cand,
                "detection": detect,
                "intervention_level": level,
                "severity": detect["severity"]
            })
        
        # 2. 정책에 따라 처리
        return self._apply_policy(candidate_scores, context)
    
    def _apply_policy(self, 
                     candidate_scores: List[Dict[str, Any]], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        정책 적용
        
        우선순위:
        1. NONE 레벨이 있으면 그 중 최선 선택
        2. SOFT만 있으면 최선 선택 (필요시 수정)
        3. 모두 HARD면 강제 개입
        """
        # NONE 레벨 후보 찾기
        none_candidates = [c for c in candidate_scores if c["intervention_level"] == InterventionLevel.NONE]
        
        if none_candidates:
            # 위반 없는 후보 중 선택 (현재는 첫 번째)
            best = none_candidates[0]
            return self._build_result(
                best["candidate"]["text"],
                best["candidate"]["id"],
                InterventionLevel.NONE,
                {"reason": "No violation detected"},
                candidate_scores
            )
        
        # SOFT 레벨 후보 찾기
        soft_candidates = [c for c in candidate_scores if c["intervention_level"] == InterventionLevel.SOFT]
        
        if soft_candidates:
            # 가장 덜 심각한 위반 선택
            best = min(soft_candidates, key=lambda x: x["severity"])
            
            # 수정 시도
            if self.enable_modification and self.llm:
                modified = self._apply_soft_modification(best, context)
                if modified:
                    return self._build_result(
                        modified,
                        best["candidate"]["id"],
                        InterventionLevel.SOFT,
                        {"reason": "Modified to reduce violation", "original": best["candidate"]["text"]},
                        candidate_scores
                    )
            
            # 수정 실패 또는 비활성화 - 그대로 선택
            return self._build_result(
                best["candidate"]["text"],
                best["candidate"]["id"],
                InterventionLevel.SOFT,
                {"reason": "Selected least severe violation", "violation": best["detection"]["top_violation"]},
                candidate_scores
            )
        
        # 모두 HARD - 강제 개입
        return self._apply_hard_intervention(candidate_scores, context)
    
    def _apply_soft_modification(self, 
                                candidate_info: Dict[str, Any], 
                                context: Dict[str, Any]) -> Optional[str]:
        """
        가벼운 수정 시도
        
        전략:
        - V1: 감정 인정 문구 추가, 단정적 표현 완화
        - V2: "if...", "it might be..." 같은 완화 표현 추가
        - V3: 직접 조언을 질문으로 전환
        """
        violation_type = candidate_info["detection"]["top_violation"]
        original_text = candidate_info["candidate"]["text"]
        
        prompt = self._build_modification_prompt(violation_type, original_text, context)
        
        try:
            modified = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
            return modified.strip()
        except Exception as e:
            print(f"Modification failed: {e}")
            return None
    
    def _build_modification_prompt(self, 
                                  violation_type: str, 
                                  original_text: str, 
                                  context: Dict[str, Any]) -> str:
        """수정 프롬프트 생성"""
        instructions = {
            "v1": "Add validation of emotions and soften any assumptions. Show empathy first.",
            "v2": "Add qualifiers like 'it might be', 'if...', or 'perhaps' to avoid stating uncertain facts.",
            "v3": "Convert direct advice into open-ended questions that help the seeker explore options."
        }
        
        instruction = instructions.get(violation_type, "Make the response more supportive and less directive.")
        
        return f"""Modify the following supporter response to reduce violations while maintaining the helpful intent.

Issue: {violation_type.upper()} - {instruction}

Original response:
{original_text}

Modified response (keep it concise and natural):"""
    
    def _apply_hard_intervention(self, 
                                candidate_scores: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        강제 개입 (V4, V5 등 고위험 위반)
        
        전략:
        1. 안전 템플릿 적용
        2. 전문가 의뢰 권장
        3. 위기 개입 프로토콜
        """
        # 가장 심각한 위반 확인
        worst = max(candidate_scores, key=lambda x: x["severity"])
        violation = worst["detection"]["top_violation"]
        
        # 안전 템플릿 선택
        if violation in ["v4", "v5"]:
            # 자살/자해 위험 관련
            safe_response = self._get_safety_template("crisis")
        else:
            # 기타 고위험
            safe_response = self._get_safety_template("general")
        
        return self._build_result(
            safe_response,
            -1,  # 템플릿 사용
            InterventionLevel.HARD,
            {
                "reason": "Critical violation detected - safety template applied",
                "detected_violation": violation,
                "all_candidates_rejected": True
            },
            candidate_scores
        )
    
    def _get_safety_template(self, template_type: str) -> str:
        """안전 템플릿"""
        templates = {
            "crisis": (
                "I'm really concerned about what you're sharing. These feelings are very serious, "
                "and I think it's important to talk to a professional who can provide proper support. "
                "Would you be open to reaching out to a counselor or a crisis helpline? "
                "You don't have to go through this alone."
            ),
            "general": (
                "I want to support you, but I think this might be something where professional guidance "
                "would be really helpful. Have you considered talking to a counselor or therapist about this?"
            )
        }
        return templates.get(template_type, templates["general"])
    
    def _build_result(self, 
                     final_response: str, 
                     candidate_id: int, 
                     intervention_level: InterventionLevel,
                     intervention_details: Dict[str, Any],
                     all_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 구성"""
        return {
            "final_response": final_response,
            "selected_candidate_id": candidate_id,
            "intervention_level": intervention_level.value,
            "intervention_details": intervention_details,
            "all_candidates": [c["candidate"] for c in all_candidates],
            "all_detections": [c["detection"] for c in all_candidates]
        }
