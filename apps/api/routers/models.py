"""
apps/api/routers/models.py
Model metadata endpoint: GET /v1/models/current
"""
from fastapi import APIRouter, HTTPException, Request, status
from schemas.response import ModelInfoResponse

router = APIRouter()


@router.get(
    "/models/current",
    response_model=ModelInfoResponse,
    summary="Champion model metadata",
    description="Returns metadata and frozen evaluation metrics for the current champion model.",
    responses={503: {"description": "No champion model loaded"}},
)
async def get_current_model(request: Request) -> ModelInfoResponse:
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None or not loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_unavailable"},
        )

    meta = loader.model_metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "churn_predictor"),
        model_version=meta["model_version"],
        algorithm=meta.get("algorithm", "unknown"),
        stage=meta.get("stage", "champion"),
        feature_set_version=meta.get("feature_set_version", "unknown"),
        pr_auc=meta.get("pr_auc"),
        roc_auc=meta.get("roc_auc"),
        brier_score=meta.get("brier_score"),
        recall_at_threshold=meta.get("recall_at_threshold"),
        precision_at_threshold=meta.get("precision_at_threshold"),
        f1_at_threshold=meta.get("f1_at_threshold"),
        threshold=meta.get("threshold"),
        promoted_at=meta.get("promoted_at"),
        promoted_by=meta.get("promoted_by"),
    )
