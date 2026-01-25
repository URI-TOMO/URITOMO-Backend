import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from sqlalchemy import select, and_

from app.core.deps import SessionDep
from app.core.token import CurrentUserDep
from app.core.errors import AppError
from app.models.room import Room, RoomLiveSession, RoomMember, RoomLiveSessionMember
from app.models.user import User
from app.meeting.schemas import SuccessResponse

router = APIRouter(prefix="/meeting", tags=["meetings"])

@router.post("/{room_id}/live-sessions/{session_id}", response_model=SuccessResponse)
async def enter_live_session(
    room_id: str,
    session_id: str,
    current_user_id: CurrentUserDep,
    session: SessionDep
):
    """
    라이브 세션에 입장합니다 (생성 및 참가 통합).
    세션이 존재하지 않으면 새로 생성하고, 존재하면 해당 세션에 참가합니다.
    """
    try:
        # 1. Fetch User
        user_result = await session.execute(select(User).where(User.id == current_user_id))
        user = user_result.scalar_one_or_none()
        if not user:
             raise AppError(status_code=401, code="40102", message="Unauthorized")

        # 2. Check Room and Membership
        room_result = await session.execute(select(Room).where(Room.id == room_id))
        room = room_result.scalar_one_or_none()
        if not room:
             raise AppError(status_code=404, code="40401", message="Room not found")

        member_result = await session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id
            )
        )
        member = member_result.scalar_one_or_none()
        if not member:
             raise AppError(status_code=403, code="40301", message="Not a member of this room")

        # 3. Create or Update Live Session
        session_result = await session.execute(select(RoomLiveSession).where(RoomLiveSession.id == session_id))
        live_session = session_result.scalar_one_or_none()
        
        if not live_session:
            # First person to enter creates the session
            live_session = RoomLiveSession(
                id=session_id,
                room_id=room_id,
                title=f"{room.title} - Session",
                status="active",
                started_by=current_user_id,
                started_at=datetime.utcnow(),
            )
            session.add(live_session)
        else:
            # If session exists but ended, reactivate it (optional, depends on policy)
            if live_session.status != "active":
                live_session.status = "active"
                live_session.ended_at = None

        # 4. Add or Update session member (Join)
        part_result = await session.execute(
            select(RoomLiveSessionMember).where(
                RoomLiveSessionMember.session_id == session_id,
                RoomLiveSessionMember.user_id == current_user_id
            )
        )
        existing_member = part_result.scalar_one_or_none()
        
        if not existing_member:
            session_member = RoomLiveSessionMember(
                id=f"lsm_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                room_id=room_id,
                member_id=member.id,
                user_id=current_user_id,
                display_name=member.display_name,
                role=member.role,
                joined_at=datetime.utcnow(),
            )
            session.add(session_member)
        else:
            # Already a member (possibly left and coming back)
            existing_member.left_at = None
            existing_member.joined_at = datetime.utcnow() # Update last join time
        
        await session.commit()
        return SuccessResponse(status="success")

    except AppError:
        raise
    except Exception as e:
        print(f"Error entering live session: {e}")
        raise AppError(status_code=500, code="50001", message="Internal server error")

    except AppError:
        raise
    except Exception as e:
        print(f"Error joining live session: {e}")
        raise AppError(status_code=500, code="50001", message="Internal server error")


@router.post("/{room_id}/live-sessions/{session_id}/leave", response_model=SuccessResponse)
async def leave_live_session(
    room_id: str,
    session_id: str,
    current_user_id: CurrentUserDep,
    session: SessionDep
):
    """
    현재 참가 중인 라이브 세션에서 나갑니다.
    """
    try:
        # 1. Check Room Membership
        member_result = await session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id
            )
        )
        member = member_result.scalar_one_or_none()
        
        if not member:
             raise AppError(status_code=403, code="40301", message="Not a member of this room")

        # 2. Find active participation
        participant_result = await session.execute(
            select(RoomLiveSessionMember).where(
                and_(
                    RoomLiveSessionMember.session_id == session_id,
                    RoomLiveSessionMember.member_id == member.id,
                    RoomLiveSessionMember.left_at.is_(None)
                )
            )
        )
        participant = participant_result.scalar_one_or_none()
        
        if not participant:
             raise AppError(status_code=404, code="40403", message="Not currently in this session")

        # 3. Update left_at timestamp
        participant.left_at = datetime.utcnow()
        await session.commit()
        await session.refresh(participant)
        
        return SuccessResponse(
            status="success",
            data={
                "message": "Left successfully",
                "participant": {
                    "id": participant.id,
                    "member_id": participant.member_id,
                    "display_name": participant.display_name,
                    "joined_at": participant.joined_at,
                    "left_at": participant.left_at
                }
            }
        )

    except AppError:
        raise
    except Exception as e:
        print(f"Error leaving live session: {e}")
        raise AppError(status_code=500, code="50001", message="Internal server error")
