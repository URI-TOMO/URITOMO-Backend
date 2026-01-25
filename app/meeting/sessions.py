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

@router.post("/{room_id}/live-sessions", response_model=SuccessResponse)
async def start_live_session(
    room_id: str,
    current_user_id: CurrentUserDep,
    session: SessionDep
):
    """
    새로운 라이브 세션을 생성합니다.
    생성자는 자동으로 첫 번째 참가자로 등록됩니다.
    """
    try:
        # 1. Fetch User (for display_name)
        user_result = await session.execute(select(User).where(User.id == current_user_id))
        user = user_result.scalar_one_or_none()
        
        if not user:
             raise AppError(status_code=401, code="40102", message="Unauthorized")

        # 2. Check Room and Membership
        room_result = await session.execute(select(Room).where(Room.id == room_id))
        room = room_result.scalar_one_or_none()
        
        if not room:
             raise AppError(status_code=404, code="40401", message="Room not found")

        # Check Membership
        member_result = await session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id
            )
        )
        member = member_result.scalar_one_or_none()
        
        if not member:
             raise AppError(status_code=403, code="40301", message="Not a member of this room")

        # 3. Create Live Session
        session_id = f"ls_{uuid.uuid4().hex[:16]}"
        session_title = user.display_name
        
        new_session = RoomLiveSession(
            id=session_id,
            room_id=room_id,
            title=session_title,
            status="active",
            started_by=current_user_id,
            started_at=datetime.utcnow(),
            ended_at=None
        )
        
        session.add(new_session)
        
        # 4. Add creator as first participant
        participant_id = f"lsm_{uuid.uuid4().hex[:16]}"
        session_member = RoomLiveSessionMember(
            id=participant_id,
            session_id=session_id,
            room_id=room_id,
            member_id=member.id,
            user_id=current_user_id,
            display_name=member.display_name,
            role=member.role,
            joined_at=datetime.utcnow(),
            left_at=None
        )
        
        session.add(session_member)
        await session.commit()
        await session.refresh(new_session)
        await session.refresh(session_member)
        
        return SuccessResponse(
            status="success",
            data={
                "session": {
                    "id": new_session.id,
                    "room_id": new_session.room_id,
                    "title": new_session.title,
                    "status": new_session.status,
                    "started_by": new_session.started_by,
                    "started_at": new_session.started_at,
                    "ended_at": new_session.ended_at
                },
                "participant": {
                    "id": session_member.id,
                    "member_id": session_member.member_id,
                    "display_name": session_member.display_name,
                    "role": session_member.role,
                    "joined_at": session_member.joined_at
                }
            }
        )

    except AppError:
        raise
    except Exception as e:
        print(f"Error starting live session: {e}")
        raise AppError(status_code=500, code="50001", message="Internal server error")


@router.post("/{room_id}/live-sessions/{session_id}/join", response_model=SuccessResponse)
async def join_live_session(
    room_id: str,
    session_id: str,
    current_user_id: CurrentUserDep,
    session: SessionDep
):
    """
    기존 활성 라이브 세션에 참가합니다.
    """
    try:
        # 1. Fetch User
        user_result = await session.execute(select(User).where(User.id == current_user_id))
        user = user_result.scalar_one_or_none()
        
        if not user:
             raise AppError(status_code=401, code="40102", message="Unauthorized")

        # 2. Check Room Membership
        member_result = await session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id
            )
        )
        member = member_result.scalar_one_or_none()
        
        if not member:
             raise AppError(status_code=403, code="40301", message="Not a member of this room")

        # 3. Check Live Session exists and is active
        live_session_result = await session.execute(
            select(RoomLiveSession).where(
                RoomLiveSession.id == session_id,
                RoomLiveSession.room_id == room_id
            )
        )
        live_session = live_session_result.scalar_one_or_none()
        
        if not live_session:
             raise AppError(status_code=404, code="40402", message="Live session not found")
        
        if live_session.status != "active":
             raise AppError(status_code=400, code="40001", message="Live session is not active")

        # 4. Check if already joined (and not left)
        existing_participant_result = await session.execute(
            select(RoomLiveSessionMember).where(
                and_(
                    RoomLiveSessionMember.session_id == session_id,
                    RoomLiveSessionMember.member_id == member.id,
                    RoomLiveSessionMember.left_at.is_(None)
                )
            )
        )
        existing_participant = existing_participant_result.scalar_one_or_none()
        
        if existing_participant:
            # Already in session, return existing participation
            return SuccessResponse(
                status="success",
                data={
                    "message": "Already joined",
                    "participant": {
                        "id": existing_participant.id,
                        "member_id": existing_participant.member_id,
                        "display_name": existing_participant.display_name,
                        "role": existing_participant.role,
                        "joined_at": existing_participant.joined_at
                    }
                }
            )

        # 5. Add as participant
        participant_id = f"lsm_{uuid.uuid4().hex[:16]}"
        session_member = RoomLiveSessionMember(
            id=participant_id,
            session_id=session_id,
            room_id=room_id,
            member_id=member.id,
            user_id=current_user_id,
            display_name=member.display_name,
            role=member.role,
            joined_at=datetime.utcnow(),
            left_at=None
        )
        
        session.add(session_member)
        await session.commit()
        await session.refresh(session_member)
        
        return SuccessResponse(
            status="success",
            data={
                "message": "Joined successfully",
                "participant": {
                    "id": session_member.id,
                    "member_id": session_member.member_id,
                    "display_name": session_member.display_name,
                    "role": session_member.role,
                    "joined_at": session_member.joined_at
                }
            }
        )

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
