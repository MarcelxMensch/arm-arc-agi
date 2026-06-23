"""
Recursive reasoning models: HRM, TRM variants, Transformer baseline.
Load via load_model_class with identifiers such as:
  - recursive_reasoning.hrm@HierarchicalReasoningModel_ACTV1
  - recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1
  - recursive_reasoning.trm_abstraction@AbstractionReasoningModel_ACTV1
  - recursive_reasoning.transformers_baseline@Model_ACTV2
"""
from utils.models.recursive_reasoning.hrm import HierarchicalReasoningModel_ACTV1
from utils.models.recursive_reasoning.transformers_baseline import Model_ACTV2
from utils.models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from utils.models.recursive_reasoning.trm_abstraction import (
    AbstractionReasoningModel_ACTV1,
)

__all__ = [
    "HierarchicalReasoningModel_ACTV1",
    "Model_ACTV2",
    "TinyRecursiveReasoningModel_ACTV1",
    "AbstractionReasoningModel_ACTV1",
]
