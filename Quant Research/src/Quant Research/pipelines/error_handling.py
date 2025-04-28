# Add to pipelines module

from prefect.tasks.notifications import SlackTask
from prefect.engine import signals

# Configure notifications
slack_alert = SlackTask(
    webhook_secret="SLACK_WEBHOOK_URL", 
    message="Pipeline {flow_name} {message}"
)

# Add to pipeline flow
@task(name="handle_errors", 
      trigger=prefect.triggers.all_failed)
def handle_pipeline_failure(task_states):
    """Handle pipeline failures and send notifications"""
    failed_tasks = [t for t in task_states if t.is_failed()]
    error_details = "\n".join([f"{t.task_name}: {t.message}" for t in failed_tasks])
    
    # Send notification
    slack_alert(message=f"Failed with errors:\n{error_details}")
    
    # Re-raise for workflow engine
    raise signals.FAIL(f"Pipeline failed with {len(failed_tasks)} task failures")