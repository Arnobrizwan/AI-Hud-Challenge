"""
Airflow DAG for ML Training Pipeline
Production-grade ML training pipeline with comprehensive orchestration
"""

from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

# Default arguments
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email": ["mlops-alerts@company.com"],
    "catchup": False,
}


def create_ml_training_dag(pipeline_config: Dict[str, Any]) -> DAG:
    """Create ML training DAG based on pipeline configuration"""

    dag_id = f"ml_training_{pipeline_config['model_name'].lower().replace(' ', '_')}"

    dag = DAG(
        dag_id=dag_id,
        description=f"Training pipeline for {pipeline_config['model_name']}",
        schedule_interval=pipeline_config.get("schedule_interval", None),
        start_date=datetime(2023, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["ml-training", pipeline_config["model_name"], "production"],
        default_args=default_args,
    )

    # Task 1: Environment Setup
    setup_environment = BashOperator(
        task_id="setup_environment",
        bash_command="""
        echo "Setting up ML training environment..."
        pip install -r requirements.txt
        echo "Environment setup completed"
        """,
        dag=dag,
    )

    # Task 2: Data Validation
    with TaskGroup("data_validation", dag=dag) as data_validation_group:

        # Check data availability
        check_data_availability = FileSensor(
            task_id="check_data_availability",
            filepath=pipeline_config.get("data_path", "/data/training_data.csv"),
            timeout=300,
            poke_interval=30,
            dag=dag,
        )

        # Validate data quality
        validate_data_quality = PythonOperator(
            task_id="validate_data_quality",
            python_callable=validate_training_data,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        # Check data schema
        check_data_schema = PythonOperator(
            task_id="check_data_schema",
            python_callable=check_data_schema,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        check_data_availability >> validate_data_quality >> check_data_schema

    # Task 3: Feature Engineering
    with TaskGroup("feature_engineering", dag=dag) as feature_engineering_group:

        # Extract features
        extract_features = PythonOperator(
            task_id="extract_features",
            python_callable=extract_features,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        # Transform features
        transform_features = PythonOperator(
            task_id="transform_features",
            python_callable=transform_features,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        # Validate features
        validate_features = PythonOperator(
            task_id="validate_features",
            python_callable=validate_features,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        extract_features >> transform_features >> validate_features

    # Task 4: Model Training
    with TaskGroup("model_training", dag=dag) as model_training_group:

        # Hyperparameter tuning (if enabled)
        if pipeline_config.get("enable_hyperparameter_tuning", False):
            hyperparameter_tuning = PythonOperator(
                task_id="hyperparameter_tuning",
                python_callable=run_hyperparameter_tuning,
                op_kwargs={"config": pipeline_config},
                pool="training_pool",
                dag=dag,
            )

        # Train model
        train_model = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            op_kwargs={"config": pipeline_config},
            pool="training_pool",
            dag=dag,
        )

        # Save model artifacts
        save_model = PythonOperator(
            task_id="save_model",
            python_callable=save_model_artifacts,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        if pipeline_config.get("enable_hyperparameter_tuning", False):
            hyperparameter_tuning >> train_model >> save_model
        else:
            train_model >> save_model

    # Task 5: Model Evaluation
    with TaskGroup("model_evaluation", dag=dag) as model_evaluation_group:

        # Evaluate model
        evaluate_model = PythonOperator(
            task_id="evaluate_model",
            python_callable=evaluate_model,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        # Generate evaluation report
        generate_evaluation_report = PythonOperator(
            task_id="generate_evaluation_report",
            python_callable=generate_evaluation_report,
            op_kwargs={"config": pipeline_config},
            dag=dag,
        )

        evaluate_model >> generate_evaluation_report

    # Task 6: Model Validation
    model_validation = PythonOperator(
        task_id="model_validation",
        python_callable=validate_model_quality,
        op_kwargs={"config": pipeline_config},
        dag=dag,
    )

    # Task 7: Model Registration
    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_in_registry,
        op_kwargs={"config": pipeline_config},
        trigger_rule="all_success",
        dag=dag,
    )

    # Task 8: Model Deployment (if enabled)
    if pipeline_config.get("include_deployment", False):
        deploy_model = PythonOperator(
            task_id="deploy_model",
            python_callable=deploy_model_to_serving,
            op_kwargs={"config": pipeline_config},
            trigger_rule="all_success",
            dag=dag,
        )

    # Task 9: Setup Monitoring (if enabled)
    if pipeline_config.get("include_monitoring", False):
        setup_monitoring = PythonOperator(
            task_id="setup_monitoring",
            python_callable=setup_model_monitoring,
            op_kwargs={"config": pipeline_config},
            trigger_rule="all_success",
            dag=dag,
        )

    # Task 10: Cleanup
    cleanup = BashOperator(
        task_id="cleanup",
        bash_command="""
        echo "Cleaning up temporary files..."
        rm -rf /tmp/ml_training_*
        echo "Cleanup completed"
        """,
        trigger_rule="all_done",
        dag=dag,
    )

    # Set up task dependencies
    setup_environment >> data_validation_group
    data_validation_group >> feature_engineering_group
    feature_engineering_group >> model_training_group
    model_training_group >> model_evaluation_group
    model_evaluation_group >> model_validation
    model_validation >> register_model

    if pipeline_config.get("include_deployment", False):
        register_model >> deploy_model
        if pipeline_config.get("include_monitoring", False):
            deploy_model >> setup_monitoring
    elif pipeline_config.get("include_monitoring", False):
        register_model >> setup_monitoring

    # Add cleanup to all paths
    if pipeline_config.get("include_deployment", False) and pipeline_config.get(
        "include_monitoring", False
    ):
        setup_monitoring >> cleanup
    elif pipeline_config.get("include_deployment", False):
        deploy_model >> cleanup
    elif pipeline_config.get("include_monitoring", False):
        setup_monitoring >> cleanup
    else:
        register_model >> cleanup

    return dag


# Task functions
def validate_training_data(config: Dict[str, Any]) -> None:
    """Validate training data quality"""
    import numpy as np
    import pandas as pd
    from mlops_orchestration_service.src.utils.data_validation import DataValidator

    data_path = config.get("data_path", "/data/training_data.csv")
    data = pd.read_csv(data_path)

    validator = DataValidator()
    validation_result = validator.validate_data(data, config.get("validation_rules", {}))

    if not validation_result.is_valid:
        raise ValueError(f"Data validation failed: {validation_result.errors}")

    print(f"Data validation passed. Shape: {data.shape}")


def check_data_schema(config: Dict[str, Any]) -> None:
    """Check data schema compliance"""
    import pandas as pd

    data_path = config.get("data_path", "/data/training_data.csv")
    data = pd.read_csv(data_path)

    expected_columns = config.get("expected_columns", [])
    if expected_columns and not all(col in data.columns for col in expected_columns):
        missing_cols = set(expected_columns) - set(data.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("Data schema validation passed")


def extract_features(config: Dict[str, Any]) -> None:
    """Extract features from raw data"""
    import pandas as pd
    from mlops_orchestration_service.src.feature_store.feature_extractor import FeatureExtractor

    data_path = config.get("data_path", "/data/training_data.csv")
    data = pd.read_csv(data_path)

    extractor = FeatureExtractor()
    features = extractor.extract_features(data, config.get("feature_definitions", []))

    # Save features
    features_path = f"/tmp/features_{config['model_name']}.csv"
    features.to_csv(features_path, index=False)

    print(f"Features extracted and saved to {features_path}")


def transform_features(config: Dict[str, Any]) -> None:
    """Transform features"""
    import pandas as pd
    from mlops_orchestration_service.src.feature_store.feature_transformer import FeatureTransformer

    features_path = f"/tmp/features_{config['model_name']}.csv"
    features = pd.read_csv(features_path)

    transformer = FeatureTransformer()
    transformed_features = transformer.transform_features(
        features, config.get("transformation_pipeline", [])
    )

    # Save transformed features
    transformed_path = f"/tmp/transformed_features_{config['model_name']}.csv"
    transformed_features.to_csv(transformed_path, index=False)

    print(f"Features transformed and saved to {transformed_path}")


def validate_features(config: Dict[str, Any]) -> None:
    """Validate feature quality"""
    import pandas as pd

    transformed_path = f"/tmp/transformed_features_{config['model_name']}.csv"
    features = pd.read_csv(transformed_path)

    # Check for missing values
    missing_pct = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
    if missing_pct > 0.1:  # 10% threshold
        raise ValueError(f"Too many missing values: {missing_pct:.2%}")

    print("Feature validation passed")


def run_hyperparameter_tuning(config: Dict[str, Any]) -> None:
    """Run hyperparameter tuning"""
    from mlops_orchestration_service.src.training.hyperparameter_tuner import HyperparameterTuner

    tuner = HyperparameterTuner()
    best_params = tuner.optimize(
        model_class=config.get("model_class"),
        param_space=config.get("hyperparameter_space", {}),
        data_path=f"/tmp/transformed_features_{config['model_name']}.csv",
        optimization_metric=config.get("optimization_metric", "accuracy"),
    )

    # Save best parameters
    import json

    params_path = f"/tmp/best_params_{config['model_name']}.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f)

    print(f"Hyperparameter tuning completed. Best params: {best_params}")


def train_model(config: Dict[str, Any]) -> None:
    """Train the model"""
    import joblib
    import pandas as pd
    from mlops_orchestration_service.src.training.model_trainer import ModelTrainer

    # Load data
    data_path = f"/tmp/transformed_features_{config['model_name']}.csv"
    data = pd.read_csv(data_path)

    # Load best parameters if available
    best_params = {}
    try:
        import json

        params_path = f"/tmp/best_params_{config['model_name']}.json"
        with open(params_path, "r") as f:
            best_params = json.load(f)
    except FileNotFoundError:
        pass

    # Merge with default parameters
    model_params = config.get("model_params", {})
    model_params.update(best_params)

    # Train model
    trainer = ModelTrainer()
    model = trainer.train_model(
        data=data,
        model_class=config.get("model_class"),
        model_params=model_params,
        target_column=config.get("target_column", "target"),
    )

    # Save model
    model_path = f"/tmp/model_{config['model_name']}.pkl"
    joblib.dump(model, model_path)

    print(f"Model trained and saved to {model_path}")


def save_model_artifacts(config: Dict[str, Any]) -> None:
    """Save model artifacts"""
    import os
    import shutil

    model_name = config["model_name"]
    artifacts_dir = f"/tmp/model_artifacts_{model_name}"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Copy model file
    shutil.copy(f"/tmp/model_{model_name}.pkl", artifacts_dir)

    # Copy feature definitions
    if os.path.exists(f"/tmp/features_{model_name}.csv"):
        shutil.copy(f"/tmp/features_{model_name}.csv", artifacts_dir)

    print(f"Model artifacts saved to {artifacts_dir}")


def evaluate_model(config: Dict[str, Any]) -> None:
    """Evaluate model performance"""
    import joblib
    import pandas as pd
    from mlops_orchestration_service.src.training.model_evaluator import ModelEvaluator

    # Load model and data
    model_path = f"/tmp/model_{config['model_name']}.pkl"
    model = joblib.load(model_path)

    data_path = f"/tmp/transformed_features_{config['model_name']}.csv"
    data = pd.read_csv(data_path)

    # Evaluate model
    evaluator = ModelEvaluator()
    evaluation_result = evaluator.evaluate_model(
        model=model,
        data=data,
        target_column=config.get("target_column", "target"),
        metrics=config.get("evaluation_metrics", ["accuracy", "precision", "recall", "f1"]),
    )

    # Save evaluation results
    import json

    eval_path = f"/tmp/evaluation_{config['model_name']}.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation_result, f)

    print(f"Model evaluation completed. Results: {evaluation_result}")


def generate_evaluation_report(config: Dict[str, Any]) -> None:
    """Generate evaluation report"""
    import json

    from mlops_orchestration_service.src.training.report_generator import ReportGenerator

    # Load evaluation results
    eval_path = f"/tmp/evaluation_{config['model_name']}.json"
    with open(eval_path, "r") as f:
        evaluation_result = json.load(f)

    # Generate report
    report_generator = ReportGenerator()
    report = report_generator.generate_evaluation_report(
        model_name=config["model_name"], evaluation_result=evaluation_result
    )

    # Save report
    report_path = f"/tmp/evaluation_report_{config['model_name']}.html"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Evaluation report generated: {report_path}")


def validate_model_quality(config: Dict[str, Any]) -> None:
    """Validate model quality against thresholds"""
    import json

    # Load evaluation results
    eval_path = f"/tmp/evaluation_{config['model_name']}.json"
    with open(eval_path, "r") as f:
        evaluation_result = json.load(f)

    # Check quality thresholds
    quality_threshold = config.get("quality_threshold", 0.8)
    primary_metric = config.get("primary_metric", "accuracy")

    if evaluation_result.get(primary_metric, 0) < quality_threshold:
        raise ValueError(
            f"Model quality below threshold: {evaluation_result.get(primary_metric, 0)} < {quality_threshold}"
        )

    print("Model quality validation passed")


def register_model_in_registry(config: Dict[str, Any]) -> None:
    """Register model in model registry"""
    from mlops_orchestration_service.src.registry.model_registry import ModelRegistry

    registry = ModelRegistry()

    model_info = {
        "model_name": config["model_name"],
        "model_path": f"/tmp/model_{config['model_name']}.pkl",
        "artifacts_path": f"/tmp/model_artifacts_{config['model_name']}",
        "evaluation_path": f"/tmp/evaluation_{config['model_name']}.json",
        "version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now().isoformat(),
    }

    registry.register_model(model_info)
    print(f"Model registered: {model_info['version']}")


def deploy_model_to_serving(config: Dict[str, Any]) -> None:
    """Deploy model to serving endpoint"""
    from mlops_orchestration_service.src.deployment.deployment_manager import ModelDeploymentManager

    deployment_manager = ModelDeploymentManager()

    deployment_config = {
        "model_name": config["model_name"],
        "model_version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "strategy": config.get("deployment_strategy", "standard"),
        "environment": config.get("environment", "production"),
    }

    deployment_result = deployment_manager.deploy_model(deployment_config)
    print(f"Model deployed: {deployment_result['endpoint_url']}")


def setup_model_monitoring(config: Dict[str, Any]) -> None:
    """Setup model monitoring"""
    from mlops_orchestration_service.src.monitoring.monitoring_service import ModelMonitoringService

    monitoring_service = ModelMonitoringService()

    monitoring_config = {
        "model_name": config["model_name"],
        "monitoring_interval_seconds": 60,
        "alert_rules": config.get("alert_rules", []),
    }

    monitoring_service.setup_model_monitoring(monitoring_config)
    print("Model monitoring setup completed")


# Create DAG instance
pipeline_config = {
    "model_name": "customer_churn_prediction",
    "data_path": "/data/customer_data.csv",
    "model_class": "sklearn.ensemble.RandomForestClassifier",
    "model_params": {"n_estimators": 100, "random_state": 42},
    "target_column": "churn",
    "enable_hyperparameter_tuning": True,
    "hyperparameter_space": {
        "n_estimators": {"type": "int", "low": 50, "high": 200},
        "max_depth": {"type": "int", "low": 3, "high": 20},
    },
    "quality_threshold": 0.85,
    "primary_metric": "accuracy",
    "include_deployment": True,
    "include_monitoring": True,
    "deployment_strategy": "canary",
}

dag = create_ml_training_dag(pipeline_config)
