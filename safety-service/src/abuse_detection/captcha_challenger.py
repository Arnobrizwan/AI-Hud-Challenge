"""
CAPTCHA Challenger
CAPTCHA-based challenge system for abuse prevention
"""

import asyncio
import hashlib
import json
import logging
import secrets
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_abuse_config

logger = logging.getLogger(__name__)


class CaptchaChallenge:
    """Individual CAPTCHA challenge"""

    def __init__(
            self,
            challenge_id: str,
            challenge_type: str,
            difficulty: str):
        self.challenge_id = challenge_id
        self.challenge_type = challenge_type
        self.difficulty = difficulty
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(minutes=10)
        self.attempts = 0
        self.max_attempts = 3
        self.is_solved = False
        self.solution = None
        self.challenge_data = {}

    def is_expired(self) -> bool:
        """Check if challenge is expired"""
        return datetime.utcnow() > self.expires_at

    def can_attempt(self) -> bool:
        """Check if challenge can be attempted"""
        return not self.is_expired() and self.attempts < self.max_attempts and not self.is_solved


class CaptchaChallenger:
    """CAPTCHA challenge system for abuse prevention"""

    def __init__(self):
        self.config = get_abuse_config()
        self.is_initialized = False

        # Challenge storage
        self.active_challenges: Dict[str, CaptchaChallenge] = {}
        self.challenge_history: Dict[str, List[Dict[str, Any]]] = {}

        # Challenge types and their configurations
        self.challenge_types = {
            "text_captcha": {
                "name": "Text CAPTCHA",
                "description": "Solve a text-based challenge",
                "difficulties": ["easy", "medium", "hard"],
            },
            "math_captcha": {
                "name": "Math CAPTCHA",
                "description": "Solve a mathematical equation",
                "difficulties": ["easy", "medium", "hard"],
            },
            "image_captcha": {
                "name": "Image CAPTCHA",
                "description": "Identify objects in images",
                "difficulties": ["easy", "medium", "hard"],
            },
            "puzzle_captcha": {
                "name": "Puzzle CAPTCHA",
                "description": "Complete a puzzle",
                "difficulties": ["easy", "medium", "hard"],
            },
        }

        # Difficulty configurations
        self.difficulty_configs = {
            # 5 minutes
            "easy": {"time_limit": 300, "max_attempts": 5, "complexity": 1},
            # 3 minutes
            "medium": {"time_limit": 180, "max_attempts": 3, "complexity": 2},
            # 2 minutes
            "hard": {"time_limit": 120, "max_attempts": 2, "complexity": 3},
        }

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the CAPTCHA challenger"""
        try:
            # Start cleanup task
            asyncio.create_task(self.cleanup_expired_challenges())

            self.is_initialized = True
            logger.info("CAPTCHA challenger initialized")

        except Exception as e:
            logger.error(f"Failed to initialize CAPTCHA challenger: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
            self.active_challenges.clear()
            self.challenge_history.clear()

            self.is_initialized = False
            logger.info("CAPTCHA challenger cleanup completed")

        except Exception as e:
            logger.error(f"Error during CAPTCHA challenger cleanup: {str(e)}")

    async def challenge_user(self,
                             user_id: str,
                             challenge_type: str = "text_captcha",
                             difficulty: str = "medium") -> Dict[str, Any]:
        """Create a CAPTCHA challenge for a user"""
        if not self.is_initialized:
            raise RuntimeError("CAPTCHA challenger not initialized")

        try:
            # Check if user already has an active challenge
            existing_challenge = self.get_active_challenge(user_id)
            if existing_challenge and existing_challenge.can_attempt():
                return {
                    "challenge_id": existing_challenge.challenge_id,
                    "challenge_type": existing_challenge.challenge_type,
                    "difficulty": existing_challenge.difficulty,
                    "challenge_data": existing_challenge.challenge_data,
                    "expires_at": existing_challenge.expires_at.isoformat(),
                    "attempts_remaining": existing_challenge.max_attempts
                    - existing_challenge.attempts,
                }

            # Generate new challenge
            challenge_id = self.generate_challenge_id()
            challenge = CaptchaChallenge(
                challenge_id, challenge_type, difficulty)

            # Generate challenge content
            challenge_data = await self.generate_challenge_content(challenge_type, difficulty)
            challenge.challenge_data = challenge_data
            challenge.solution = challenge_data.get("solution")

            # Store challenge
            self.active_challenges[user_id] = challenge

            # Record in history
            if user_id not in self.challenge_history:
                self.challenge_history[user_id] = []

            self.challenge_history[user_id].append(
                {
                    "challenge_id": challenge_id,
                    "challenge_type": challenge_type,
                    "difficulty": difficulty,
                    "created_at": challenge.created_at.isoformat(),
                    "status": "created",
                }
            )

            logger.info(
                f"Created {difficulty} {challenge_type} challenge for user {user_id}")

            return {
                "challenge_id": challenge_id,
                "challenge_type": challenge_type,
                "difficulty": difficulty,
                "challenge_data": challenge_data,
                "expires_at": challenge.expires_at.isoformat(),
                "attempts_remaining": challenge.max_attempts,
            }

        except Exception as e:
            logger.error(
                f"Failed to create challenge for user {user_id}: {str(e)}")
            raise

    async def verify_challenge(
        self, user_id: str, challenge_id: str, user_response: str
    ) -> Dict[str, Any]:
        """Verify a user's response to a CAPTCHA challenge"""
        if not self.is_initialized:
            raise RuntimeError("CAPTCHA challenger not initialized")

        try:
            # Get active challenge
            challenge = self.get_active_challenge(user_id)

            if not challenge:
                return {
                    "success": False,
                    "error": "No active challenge found",
                    "can_retry": False}

            if challenge.challenge_id != challenge_id:
                return {
                    "success": False,
                    "error": "Invalid challenge ID",
                    "can_retry": True}

            if not challenge.can_attempt():
                return {
                    "success": False,
                    "error": "Challenge expired or max attempts reached",
                    "can_retry": False,
                }

            # Increment attempts
            challenge.attempts += 1

            # Verify response
            is_correct = await self.verify_response(challenge, user_response)

            if is_correct:
                challenge.is_solved = True

                # Update history
                self.update_challenge_history(user_id, challenge_id, "solved")

                # Remove from active challenges
                del self.active_challenges[user_id]

                logger.info(f"User {user_id} solved challenge {challenge_id}")

                return {
                    "success": True,
                    "message": "Challenge solved successfully",
                    "can_retry": False,
                }
            else:
                # Check if user can retry
                can_retry = challenge.can_attempt()

                if not can_retry:
                    # Update history
                    self.update_challenge_history(
                        user_id, challenge_id, "failed")

                    # Remove from active challenges
                    del self.active_challenges[user_id]

                logger.info(
                    f"User {user_id} failed challenge {challenge_id} (attempt {challenge.attempts})"
                )

                return {
                    "success": False,
                    "error": "Incorrect response",
                    "can_retry": can_retry,
                    "attempts_remaining": challenge.max_attempts -
                    challenge.attempts,
                }

        except Exception as e:
            logger.error(
                f"Failed to verify challenge for user {user_id}: {str(e)}")
            return {
                "success": False,
                "error": "Verification failed",
                "can_retry": False}

    def get_active_challenge(self, user_id: str) -> Optional[CaptchaChallenge]:
        """Get active challenge for a user"""
        challenge = self.active_challenges.get(user_id)

        if challenge and challenge.is_expired():
            del self.active_challenges[user_id]
            return None

        return challenge

    async def generate_challenge_content(
        self, challenge_type: str, difficulty: str
    ) -> Dict[str, Any]:
        """Generate challenge content based on type and difficulty"""
        try:
            config = self.difficulty_configs.get(
                difficulty, self.difficulty_configs["medium"])

            if challenge_type == "text_captcha":
                return await self.generate_text_captcha(config)
            elif challenge_type == "math_captcha":
                return await self.generate_math_captcha(config)
            elif challenge_type == "image_captcha":
                return await self.generate_image_captcha(config)
            elif challenge_type == "puzzle_captcha":
                return await self.generate_puzzle_captcha(config)
            else:
                raise ValueError(f"Unknown challenge type: {challenge_type}")

        except Exception as e:
            logger.error(f"Failed to generate challenge content: {str(e)}")
            raise

    async def generate_text_captcha(
            self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text-based CAPTCHA"""
        try:
            complexity = config["complexity"]

            if complexity == 1:  # Easy
                length = 4
                chars = string.ascii_uppercase + string.digits
            elif complexity == 2:  # Medium
                length = 6
                chars = string.ascii_uppercase + string.digits + string.ascii_lowercase
            else:  # Hard
                length = 8
                chars = string.ascii_letters + string.digits + "!@#$%^&*"

            # Generate random text
            text = "".join(secrets.choice(chars) for _ in range(length))

            # Add some distortion (simplified)
            distorted_text = self.distort_text(text)

            return {
                "type": "text_captcha",
                "question": "Enter the text you see in the image:",
                "image_data": distorted_text,  # In real implementation, this would be base64 image
                "solution": text,
                "hint": f"Enter {length} characters",
            }

        except Exception as e:
            logger.error(f"Text CAPTCHA generation failed: {str(e)}")
            raise

    async def generate_math_captcha(
            self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate math-based CAPTCHA"""
        try:
            complexity = config["complexity"]

            if complexity == 1:  # Easy
                a = secrets.randbelow(10) + 1
                b = secrets.randbelow(10) + 1
                operation = secrets.choice(["+", "-"])
            elif complexity == 2:  # Medium
                a = secrets.randbelow(20) + 1
                b = secrets.randbelow(20) + 1
                operation = secrets.choice(["+", "-", "*"])
            else:  # Hard
                a = secrets.randbelow(50) + 1
                b = secrets.randbelow(50) + 1
                operation = secrets.choice(["+", "-", "*", "/"])

            # Calculate answer
            if operation == "+":
                answer = a + b
                question = f"{a} + {b} = ?"
            elif operation == "-":
                answer = a - b
                question = f"{a} - {b} = ?"
            elif operation == "*":
                answer = a * b
                question = f"{a} × {b} = ?"
            else:  # division
                answer = a // b
                question = f"{a} ÷ {b} = ?"

            return {
                "type": "math_captcha",
                "question": question,
                "solution": str(answer),
                "hint": "Enter the numerical answer",
            }

        except Exception as e:
            logger.error(f"Math CAPTCHA generation failed: {str(e)}")
            raise

    async def generate_image_captcha(
            self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image-based CAPTCHA"""
        try:
            # In a real implementation, this would generate actual images
            # For now, we'll simulate with text descriptions

            objects = [
                "car",
                "tree",
                "house",
                "dog",
                "cat",
                "bird",
                "flower",
                "book"]
            complexity = config["complexity"]

            if complexity == 1:  # Easy
                target_objects = secrets.sample(objects, 1)
            elif complexity == 2:  # Medium
                target_objects = secrets.sample(objects, 2)
            else:  # Hard
                target_objects = secrets.sample(objects, 3)

            # Generate fake image description
            image_description = f"Image contains: {', '.join(objects)}"

            return {
                "type": "image_captcha",
                "question": f"Select all {len(target_objects)} objects from the image:",
                "image_data": image_description,
                "solution": ",".join(target_objects),
                "options": objects,
                "hint": f"Select {len(target_objects)} objects",
            }

        except Exception as e:
            logger.error(f"Image CAPTCHA generation failed: {str(e)}")
            raise

    async def generate_puzzle_captcha(
            self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate puzzle-based CAPTCHA"""
        try:
            complexity = config["complexity"]

            if complexity == 1:  # Easy
                puzzle_type = "sequence"
                sequence = [1, 2, 3, 4, 5]
                missing = secrets.choice(sequence)
                puzzle_data = [x if x != missing else "?" for x in sequence]
            elif complexity == 2:  # Medium
                puzzle_type = "pattern"
                pattern = ["A", "B", "C", "A", "B", "?"]
                puzzle_data = pattern
            else:  # Hard
                puzzle_type = "logic"
                puzzle_data = ["2", "4", "8", "16", "?"]

            return {
                "type": "puzzle_captcha",
                "question": f"Complete the {puzzle_type} puzzle:",
                "puzzle_data": puzzle_data,
                "solution": self.solve_puzzle(puzzle_type, puzzle_data),
                "hint": f"Find the pattern in the {puzzle_type}",
            }

        except Exception as e:
            logger.error(f"Puzzle CAPTCHA generation failed: {str(e)}")
            raise

    def solve_puzzle(self, puzzle_type: str, puzzle_data: List[str]) -> str:
        """Solve a puzzle (simplified)"""
        if puzzle_type == "sequence":
            # Find missing number in sequence
            numbers = [int(x) for x in puzzle_data if x != "?"]
            if len(numbers) >= 2:
                diff = numbers[1] - numbers[0]
                return str(numbers[-1] + diff)
        elif puzzle_type == "pattern":
            # Find next in pattern
            return "C"  # Simplified
        elif puzzle_type == "logic":
            # Find next in logic sequence
            return "32"  # Simplified (powers of 2)

        return "unknown"

    def distort_text(self, text: str) -> str:
        """Distort text for CAPTCHA (simplified)"""
        # In a real implementation, this would apply visual distortions
        # For now, we'll just add some random characters
        distorted = []
        for char in text:
            distorted.append(char)
            if secrets.randbelow(3) == 0:  # 33% chance
                distorted.append(secrets.choice("~!@#$%^&*()_+-=[]{}|;:,.<>?"))

        return "".join(distorted)

    async def verify_response(
            self,
            challenge: CaptchaChallenge,
            user_response: str) -> bool:
        """Verify user's response to a challenge"""
        try:
            if not challenge.solution:
                return False

            # Normalize responses
            user_response = user_response.strip().lower()
            solution = challenge.solution.strip().lower()

            # Check for exact match
            if user_response == solution:
                return True

            # Check for partial matches (for image CAPTCHA)
            if challenge.challenge_type == "image_captcha":
                user_objects = set(user_response.split(","))
                solution_objects = set(solution.split(","))
                return user_objects == solution_objects

            return False

        except Exception as e:
            logger.error(f"Response verification failed: {str(e)}")
            return False

    def update_challenge_history(
            self,
            user_id: str,
            challenge_id: str,
            status: str):
        """Update challenge history for a user"""
        try:
            if user_id in self.challenge_history:
                for entry in self.challenge_history[user_id]:
                    if entry["challenge_id"] == challenge_id:
                        entry["status"] = status
                        entry["completed_at"] = datetime.utcnow().isoformat()
                        break

        except Exception as e:
            logger.error(f"Failed to update challenge history: {str(e)}")

    def generate_challenge_id(self) -> str:
        """Generate a unique challenge ID"""
        return hashlib.md5(
            f"{secrets.token_hex(16)}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:16]

    async def cleanup_expired_challenges(self) -> Dict[str, Any]:
        """Background task to clean up expired challenges"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                if not self.is_initialized:
                    break

                # Remove expired challenges
                expired_users = []
                for user_id, challenge in self.active_challenges.items():
                    if challenge.is_expired():
                        expired_users.append(user_id)
                        self.update_challenge_history(
                            user_id, challenge.challenge_id, "expired")

                for user_id in expired_users:
                    del self.active_challenges[user_id]

                if expired_users:
                    logger.debug(
                        f"Cleaned up {len(expired_users)} expired challenges")

            except Exception as e:
                logger.error(f"Challenge cleanup task failed: {str(e)}")
                await asyncio.sleep(60)

    async def get_challenge_statistics(self) -> Dict[str, Any]:
        """Get CAPTCHA challenge statistics"""
        try:
            total_challenges = sum(len(history)
                                   for history in self.challenge_history.values())
            active_challenges = len(self.active_challenges)

            # Calculate success rate
            solved_count = 0
            failed_count = 0

            for history in self.challenge_history.values():
                for entry in history:
                    if entry["status"] == "solved":
                        solved_count += 1
                    elif entry["status"] in ["failed", "expired"]:
                        failed_count += 1

            total_completed = solved_count + failed_count
            success_rate = solved_count / total_completed if total_completed > 0 else 0.0

            return {
                "total_challenges": total_challenges,
                "active_challenges": active_challenges,
                "solved_challenges": solved_count,
                "failed_challenges": failed_count,
                "success_rate": success_rate,
                "challenge_types": list(self.challenge_types.keys()),
            }

        except Exception as e:
            logger.error(f"Challenge statistics calculation failed: {str(e)}")
            return {"error": str(e)}
