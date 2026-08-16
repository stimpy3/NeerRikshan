from fastapi import APIRouter

router = APIRouter(prefix="/wetlands", tags=["wetlands"])

# Placeholder data until the GEE pipeline and Supabase/PostGIS store are wired up.
_PLACEHOLDER_QUEUE = [
    {
        "id": 1,
        "name": "East Kolkata Wetlands (sector 4)",
        "area_lost_ha": 12.4,
        "has_structures": True,
        "severity": "high",
    },
    {
        "id": 2,
        "name": "Rajarhat wetland cluster",
        "area_lost_ha": 8.1,
        "has_structures": False,
        "severity": "medium",
    },
    {
        "id": 3,
        "name": "Bantala fish farms boundary",
        "area_lost_ha": 2.3,
        "has_structures": True,
        "severity": "medium",
    },
]


@router.get("")
def list_wetlands():
    """Ranked enforcement queue."""
    return _PLACEHOLDER_QUEUE


@router.get("/{wetland_id}/trend")
def get_area_trend(wetland_id: int):
    """Area-vs-time series for a single wetland."""
    return [
        {"year": "2015", "hectares": 125},
        {"year": "2017", "hectares": 119},
        {"year": "2019", "hectares": 108},
        {"year": "2021", "hectares": 97},
        {"year": "2023", "hectares": 84},
        {"year": "2025", "hectares": 71},
    ]
