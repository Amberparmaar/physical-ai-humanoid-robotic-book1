from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from schemas import User, UserCreate, Token, UserProfile, UserProfileBase
from models import User as UserModel, UserProfile as UserProfileModel
from services import auth

router = APIRouter()


@router.get("/profile", response_model=UserProfile)
def get_user_profile(
    current_user: UserModel = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """Get the current user's profile"""
    profile = db.query(UserProfileModel).filter(
        UserProfileModel.user_id == current_user.id
    ).first()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return profile


@router.put("/profile", response_model=UserProfile)
def update_user_profile(
    profile_update: UserProfileBase,
    current_user: UserModel = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """Update the current user's profile"""
    profile = db.query(UserProfileModel).filter(
        UserProfileModel.user_id == current_user.id
    ).first()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    for field, value in profile_update.dict(exclude_unset=True).items():
        setattr(profile, field, value)

    db.commit()
    db.refresh(profile)

    return profile