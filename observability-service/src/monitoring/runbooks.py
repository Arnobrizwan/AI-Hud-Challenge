"""
Automated runbooks for incident response
"""

import asyncio
import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


class RunbookStatus(Enum):
    """Runbook execution status"""

    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVAL_PENDING = "approval_pending"
    APPROVAL_DENIED = "approval_denied"
    MANUAL_INTERVENTION = "manual_intervention"


class StepType(Enum):
    """Runbook step types"""

    COMMAND = "command"
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    KUBERNETES_ACTION = "kubernetes_action"
    NOTIFICATION = "notification"
    WAIT = "wait"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    MANUAL = "manual"


class ApprovalStatus(Enum):
    """Approval status"""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class RunbookStep:
    """Individual runbook step"""

    id: str
    name: str
    description: str
    step_type: StepType
    command: Optional[str] = None
    http_config: Optional[Dict[str, Any]] = None
    db_config: Optional[Dict[str, Any]] = None
    k8s_config: Optional[Dict[str, Any]] = None
    notification_config: Optional[Dict[str, Any]] = None
    wait_duration: Optional[int] = None  # seconds
    condition: Optional[str] = None
    loop_config: Optional[Dict[str, Any]] = None
    critical: bool = True
    requires_approval: bool = False
    timeout_seconds: int = 300
    retry_count: int = 0
    retry_delay: int = 30


@dataclass
class Runbook:
    """Runbook definition"""

    id: str
    name: str
    description: str
    version: str = "1.0"
    trigger_conditions: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[RunbookStep] = field(default_factory=list)
    requires_approval: bool = False
    timeout_minutes: int = 60
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of runbook step execution"""

    step_id: str
    success: bool
    output: str
    error_message: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunbookExecution:
    """Runbook execution instance"""

    id: str
    runbook_id: str
    incident_id: Optional[str] = None
    execution_params: Dict[str, Any] = field(default_factory=dict)
    status: RunbookStatus = RunbookStatus.INITIALIZING
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    executed_by: Optional[str] = None
    step_results: List[StepResult] = field(default_factory=list)
    error_message: Optional[str] = None
    approval_status: Optional[ApprovalStatus] = None
    approval_required_by: Optional[str] = None
    approval_notes: Optional[str] = None


@dataclass
class ApprovalRequest:
    """Approval request for runbook execution"""

    id: str
    runbook_id: str
    execution_id: str
    incident_id: Optional[str]
    requested_by: str
    requested_at: datetime
    approvers: List[str]
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    notes: Optional[str] = None
    expires_at: Optional[datetime] = None


class RunbookExecutionEngine:
    """Execute runbook steps"""

    def __init__(self):
        self.execution_environment = {}
        self.active_executions = {}

    async def setup_environment(self, config: Dict[str, Any]):
        """Set up execution environment"""
        self.execution_environment = config
        logger.info("Runbook execution environment configured")

    async def execute_step(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute a single runbook step"""

        start_time = datetime.utcnow()

        try:
            if step.step_type == StepType.COMMAND:
                result = await self._execute_command(step, context)
            elif step.step_type == StepType.HTTP_REQUEST:
                result = await self._execute_http_request(step, context)
            elif step.step_type == StepType.DATABASE_QUERY:
                result = await self._execute_database_query(step, context)
            elif step.step_type == StepType.KUBERNETES_ACTION:
                result = await self._execute_kubernetes_action(step, context)
            elif step.step_type == StepType.NOTIFICATION:
                result = await self._execute_notification(step, context)
            elif step.step_type == StepType.WAIT:
                result = await self._execute_wait(step, context)
            elif step.step_type == StepType.CONDITIONAL:
                result = await self._execute_conditional(step, context)
            elif step.step_type == StepType.LOOP:
                result = await self._execute_loop(step, context)
            elif step.step_type == StepType.MANUAL:
                result = await self._execute_manual(step, context)
            else:
                result = StepResult(
                    step_id=step.id,
                    success=False,
                    output="",
                    error_message=f"Unknown step type: {step.step_type}",
                )

            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return StepResult(
                step_id=step.id,
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def _execute_command(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute shell command"""

        try:
            # Substitute variables in command
            command = self._substitute_variables(step.command, context)

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=step.timeout_seconds
            )

            output = stdout.decode("utf-8")
            error_output = stderr.decode("utf-8")

            success = process.returncode == 0

            return StepResult(
                step_id=step.id,
                success=success,
                output=output,
                error_message=error_output if not success else None,
                metadata={"return_code": process.returncode},
            )

        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.id,
                success=False,
                output="",
                error_message=f"Command timed out after {step.timeout_seconds} seconds",
            )
        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_http_request(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute HTTP request"""

        try:
            config = step.http_config
            url = self._substitute_variables(config["url"], context)
            method = config.get("method", "GET")
            headers = config.get("headers", {})
            data = config.get("data")

            # Substitute variables in headers and data
            headers = {k: self._substitute_variables(v, context) for k, v in headers.items()}
            if data:
                data = self._substitute_variables(data, context)

            # Make request
            response = requests.request(
                method=method, url=url, headers=headers, data=data, timeout=step.timeout_seconds
            )

            success = 200 <= response.status_code < 300

            return StepResult(
                step_id=step.id,
                success=success,
                output=response.text,
                error_message=f"HTTP {response.status_code}" if not success else None,
                metadata={"status_code": response.status_code, "headers": dict(response.headers)},
            )

        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_database_query(
        self, step: RunbookStep, context: Dict[str, Any]
    ) -> StepResult:
        """Execute database query"""

        try:
            config = step.db_config
            query = self._substitute_variables(config["query"], context)

            # This would integrate with actual database connection
            # For now, return mock result

            return StepResult(
                step_id=step.id,
                success=True,
                output="Query executed successfully",
                metadata={"rows_affected": 1},
            )

        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_kubernetes_action(
        self, step: RunbookStep, context: Dict[str, Any]
    ) -> StepResult:
        """Execute Kubernetes action"""

        try:
            config = step.k8s_config
            action = config["action"]
            resource = config["resource"]
            namespace = config.get("namespace", "default")

            # This would integrate with Kubernetes API
            # For now, return mock result

            return StepResult(
                step_id=step.id,
                success=True,
                output=f"Kubernetes {action} on {resource} completed",
                metadata={"namespace": namespace},
            )

        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_notification(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute notification step"""

        try:
            config = step.notification_config
            message = self._substitute_variables(config["message"], context)
            channels = config.get("channels", [])

            # This would integrate with notification system
            # For now, just log

            logger.info(f"Notification: {message} to {channels}")

            return StepResult(
                step_id=step.id,
                success=True,
                output=f"Notification sent to {len(channels)} channels",
                metadata={"channels": channels},
            )

        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_wait(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute wait step"""

        try:
            duration = step.wait_duration or 10
            await asyncio.sleep(duration)

            return StepResult(
                step_id=step.id, success=True, output=f"Waited for {duration} seconds"
            )

        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_conditional(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute conditional step"""

        try:
            condition = step.condition
            if not condition:
                return StepResult(
                    step_id=step.id,
                    success=False,
                    output="",
                    error_message="No condition specified",
                )

            # Evaluate condition (simplified)
            condition_result = self._evaluate_condition(condition, context)

            return StepResult(
                step_id=step.id,
                success=True,
                output=f"Condition evaluated: {condition_result}",
                metadata={"condition_result": condition_result},
            )

        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_loop(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute loop step"""

        try:
            config = step.loop_config
            iterations = config.get("iterations", 1)
            loop_steps = config.get("steps", [])

            results = []
            for i in range(iterations):
                loop_context = {**context, "loop_index": i}

                for loop_step in loop_steps:
                    result = await self.execute_step(loop_step, loop_context)
                    results.append(result)

                    if not result.success and loop_step.critical:
                        break

                if not all(r.success for r in results):
                    break

            return StepResult(
                step_id=step.id,
                success=all(r.success for r in results),
                output=f"Loop completed {len(results)} iterations",
                metadata={"iterations": len(results), "results": results},
            )

        except Exception as e:
            return StepResult(step_id=step.id, success=False, output="", error_message=str(e))

    async def _execute_manual(self, step: RunbookStep, context: Dict[str, Any]) -> StepResult:
        """Execute manual step (requires human intervention)"""

        # This would trigger a manual intervention workflow
        logger.info(f"Manual step {step.id} requires human intervention")

        return StepResult(
            step_id=step.id,
            success=False,
            output="Manual intervention required",
            error_message="Step requires manual intervention",
            metadata={"requires_manual_intervention": True},
        )

    def _substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Substitute variables in text with context values"""

        if not text:
            return text

        for key, value in context.items():
            placeholder = f"${{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, str(value))

        return text

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate condition string (simplified)"""

        # This is a simplified condition evaluator
        # In practice, you'd use a proper expression evaluator

        try:
            # Replace variables in condition
            condition = self._substitute_variables(condition, context)

            # Simple evaluation (not safe for production)
            return eval(condition)
        except:
            return False


class ApprovalManager:
    """Manage runbook execution approvals"""

    def __init__(self):
        self.approval_policies = {}
        self.pending_approvals = {}

    async def configure_approval_policies(self, policies: List[Dict[str, Any]]):
        """Configure approval policies"""

        for policy in policies:
            self.approval_policies[policy["id"]] = policy

    async def request_approval(
        self, runbook: Runbook, incident: Optional[Any], execution_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request approval for runbook execution"""

        # Find applicable approval policy
        policy = self._find_approval_policy(runbook, incident)

        if not policy:
            return {"approved": True, "reason": "no_approval_required"}

        # Create approval request
        approval_request = ApprovalRequest(
            id=str(uuid.uuid4()),
            runbook_id=runbook.id,
            execution_id=str(uuid.uuid4()),
            incident_id=incident.id if incident else None,
            requested_by=execution_params.get("requested_by", "system"),
            requested_at=datetime.utcnow(),
            approvers=policy.get("approvers", []),
            expires_at=datetime.utcnow() + timedelta(minutes=policy.get("timeout_minutes", 30)),
        )

        self.pending_approvals[approval_request.id] = approval_request

        # Send approval notifications
        await self._send_approval_notifications(approval_request)

        return {
            "approved": False,
            "approval_request_id": approval_request.id,
            "approvers": approval_request.approvers,
            "expires_at": approval_request.expires_at.isoformat(),
        }

    def _find_approval_policy(
        self, runbook: Runbook, incident: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """Find applicable approval policy"""

        for policy in self.approval_policies.values():
            if policy.get("runbook_ids") and runbook.id in policy["runbook_ids"]:
                return policy
            if (
                policy.get("incident_types")
                and incident
                and incident.type in policy["incident_types"]
            ):
                return policy
            if policy.get("default", False):
                return policy

        return None

    async def _send_approval_notifications(self, approval_request: ApprovalRequest):
        """Send approval notifications to approvers"""

        # This would integrate with notification system
        logger.info(
            f"Approval requested for runbook {approval_request.runbook_id} from {approval_request.requested_by}"
        )

    async def process_approval(
        self, approval_id: str, approver: str, approved: bool, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process approval decision"""

        approval = self.pending_approvals.get(approval_id)
        if not approval:
            return {"success": False, "error": "Approval not found"}

        if approval.status != ApprovalStatus.PENDING:
            return {"success": False, "error": "Approval already processed"}

        if datetime.utcnow() > approval.expires_at:
            approval.status = ApprovalStatus.EXPIRED
            return {"success": False, "error": "Approval expired"}

        approval.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        approval.approved_by = approver
        approval.approved_at = datetime.utcnow()
        approval.notes = notes

        return {
            "success": True,
            "approved": approved,
            "approver": approver,
            "approved_at": approval.approved_at.isoformat(),
        }


class RunbookRegistry:
    """Registry for runbook definitions"""

    def __init__(self):
        self.runbooks = {}

    async def store_runbook(self, runbook: Runbook):
        """Store runbook definition"""
        self.runbooks[runbook.id] = runbook
        logger.info(f"Stored runbook: {runbook.name}")

    async def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Get runbook by ID"""
        return self.runbooks.get(runbook_id)

    async def list_runbooks(self, tags: Optional[List[str]] = None) -> List[Runbook]:
        """List runbooks, optionally filtered by tags"""

        runbooks = list(self.runbooks.values())

        if tags:
            runbooks = [r for r in runbooks if any(tag in r.tags for tag in tags)]

        return runbooks

    async def search_runbooks(self, query: str) -> List[Runbook]:
        """Search runbooks by name or description"""

        query_lower = query.lower()
        return [
            r
            for r in self.runbooks.values()
            if query_lower in r.name.lower() or query_lower in r.description.lower()
        ]


class RunbookEngine:
    """Main runbook engine"""

    def __init__(self):
        self.runbook_registry = RunbookRegistry()
        self.execution_engine = RunbookExecutionEngine()
        self.approval_manager = ApprovalManager()
        self.audit_logger = AuditLogger()
        self.is_initialized = False

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize runbook system"""

        # Load runbook definitions
        await self.load_runbook_definitions(config.get("runbooks_path", "config/runbooks.json"))

        # Set up execution environment
        await self.execution_engine.setup_environment(config.get("execution_config", {}))

        # Configure approvals
        await self.approval_manager.configure_approval_policies(config.get("approval_policies", []))

        self.is_initialized = True
        logger.info("Runbook engine initialized")

    async def load_runbook_definitions(self, runbooks_path: str):
        """Load runbook definitions from configuration"""

        try:
            with open(runbooks_path, "r") as f:
                runbooks_data = json.load(f)

            for runbook_data in runbooks_data:
                runbook = self._create_runbook_from_config(runbook_data)
                await self.runbook_registry.store_runbook(runbook)

        except FileNotFoundError:
            logger.warning(f"Runbooks file not found: {runbooks_path}")
        except Exception as e:
            logger.error(f"Failed to load runbooks: {str(e)}")

    def _create_runbook_from_config(self, config: Dict[str, Any]) -> Runbook:
        """Create runbook from configuration"""

        steps = []
        for step_config in config.get("steps", []):
            step = RunbookStep(
                id=step_config["id"],
                name=step_config["name"],
                description=step_config["description"],
                step_type=StepType(step_config["type"]),
                command=step_config.get("command"),
                http_config=step_config.get("http_config"),
                db_config=step_config.get("db_config"),
                k8s_config=step_config.get("k8s_config"),
                notification_config=step_config.get("notification_config"),
                wait_duration=step_config.get("wait_duration"),
                condition=step_config.get("condition"),
                loop_config=step_config.get("loop_config"),
                critical=step_config.get("critical", True),
                requires_approval=step_config.get("requires_approval", False),
                timeout_seconds=step_config.get("timeout_seconds", 300),
                retry_count=step_config.get("retry_count", 0),
                retry_delay=step_config.get("retry_delay", 30),
            )
            steps.append(step)

        return Runbook(
            id=config["id"],
            name=config["name"],
            description=config["description"],
            version=config.get("version", "1.0"),
            trigger_conditions=config.get("trigger_conditions", []),
            steps=steps,
            requires_approval=config.get("requires_approval", False),
            timeout_minutes=config.get("timeout_minutes", 60),
            tags=config.get("tags", []),
        )

    async def execute_runbook(
        self, runbook_id: str, incident: Optional[Any], execution_params: Dict[str, Any]
    ) -> RunbookExecution:
        """Execute runbook for incident response"""

        runbook = await self.runbook_registry.get_runbook(runbook_id)
        if not runbook:
            raise ValueError(f"Runbook {runbook_id} not found")

        execution = RunbookExecution(
            id=str(uuid.uuid4()),
            runbook_id=runbook_id,
            incident_id=incident.id if incident else None,
            execution_params=execution_params,
            status=RunbookStatus.INITIALIZING,
            started_at=datetime.utcnow(),
            executed_by=execution_params.get("executed_by", "system"),
        )

        try:
            # Check if approval is required
            if runbook.requires_approval:
                approval_result = await self.approval_manager.request_approval(
                    runbook, incident, execution_params
                )

                if not approval_result["approved"]:
                    execution.status = RunbookStatus.APPROVAL_PENDING
                    execution.approval_status = ApprovalStatus.PENDING
                    return execution

            # Execute runbook steps
            execution.status = RunbookStatus.RUNNING
            step_results = []

            for step in runbook.steps:
                step_result = await self.execution_engine.execute_step(step, execution_params)
                step_results.append(step_result)

                # Check if step failed and handle accordingly
                if not step_result.success and step.critical:
                    execution.status = RunbookStatus.FAILED
                    execution.error_message = step_result.error_message
                    break

                # Check for manual intervention requirement
                if step_result.metadata.get("requires_manual_intervention"):
                    execution.status = RunbookStatus.MANUAL_INTERVENTION
                    break

            if execution.status == RunbookStatus.RUNNING:
                execution.status = RunbookStatus.COMPLETED

            execution.step_results = step_results
            execution.completed_at = datetime.utcnow()

            # Log execution for audit
            await self.audit_logger.log_runbook_execution(execution)

        except Exception as e:
            execution.status = RunbookStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(f"Runbook execution failed: {str(e)}")

        return execution

    async def create_runbook(self, runbook_definition: Dict[str, Any]) -> Runbook:
        """Create new automated runbook"""

        runbook = self._create_runbook_from_config(runbook_definition)

        # Validate runbook
        validation_result = await self.validate_runbook(runbook)
        if not validation_result["is_valid"]:
            raise ValueError(f"Invalid runbook: {validation_result['errors']}")

        # Store runbook
        await self.runbook_registry.store_runbook(runbook)

        return runbook

    async def validate_runbook(self, runbook: Runbook) -> Dict[str, Any]:
        """Validate runbook definition"""

        errors = []

        # Check required fields
        if not runbook.name:
            errors.append("Runbook name is required")

        if not runbook.steps:
            errors.append("Runbook must have at least one step")

        # Validate steps
        for i, step in enumerate(runbook.steps):
            if not step.name:
                errors.append(f"Step {i+1} name is required")

            if step.step_type == StepType.COMMAND and not step.command:
                errors.append(f"Step {i+1} command is required for command type")

            if step.step_type == StepType.HTTP_REQUEST and not step.http_config:
                errors.append(f"Step {i+1} http_config is required for http_request type")

        return {"is_valid": len(errors) == 0, "errors": errors}

    async def cleanup(self):
        """Cleanup runbook engine"""
        self.is_initialized = False
        logger.info("Runbook engine cleaned up")


class AuditLogger:
    """Audit logging for runbook executions"""

    def __init__(self):
        self.audit_logs = []

    async def log_runbook_execution(self, execution: RunbookExecution):
        """Log runbook execution for audit"""

        audit_entry = {
            "execution_id": execution.id,
            "runbook_id": execution.runbook_id,
            "incident_id": execution.incident_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "executed_by": execution.executed_by,
            "step_count": len(execution.step_results),
            "successful_steps": len([r for r in execution.step_results if r.success]),
            "failed_steps": len([r for r in execution.step_results if not r.success]),
        }

        self.audit_logs.append(audit_entry)
        logger.info(f"Audit logged: Runbook execution {execution.id}")

    async def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs"""
        return self.audit_logs[-limit:]
