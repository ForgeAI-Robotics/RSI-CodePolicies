"""Task identifiers shared by launcher, actor adapter and native evaluator."""

HEAD_CENTER_TASK = "single_left_head_center"
SINGLE_LEFT_TASKS = ("single_left", HEAD_CENTER_TASK)
TASK_VARIANTS = ("two_can", *SINGLE_LEFT_TASKS)
