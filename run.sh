#!/bin/bash

# URITOMO Backend simple runner
# This script works on Mac, Linux, and Git Bash on Windows.

# Text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting URITOMO Backend services...${NC}"

# 0. Clean removed LiveKit worker containers (if they still exist)
docker-compose rm -sf livekit_publisher livekit_sniffer >/dev/null 2>&1

# 1. Start Docker containers in background
# --build: Ensures changes to Dockerfile or app code are reflected
docker-compose up -d --build

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to start docker containers. Make sure Docker Desktop is running.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Containers are up and running.${NC}"

# 1-1. Ensure LiveKit workers are recreated with latest env
docker-compose up -d --force-recreate livekit_realtime_agent

# 1-2. Warn if WORKER_SERVICE_KEY is missing (LiveKit workers need it)
if [ -f .env ]; then
    WORKER_KEY_LINE=$(grep -E '^WORKER_SERVICE_KEY=' .env | tail -n1)
    WORKER_KEY_VALUE="${WORKER_KEY_LINE#WORKER_SERVICE_KEY=}"
    if [ -z "$WORKER_KEY_VALUE" ]; then
        echo -e "${YELLOW}⚠️  WORKER_SERVICE_KEY is empty. LiveKit workers may fail to authenticate.${NC}"
    fi
fi

LAN_IP=""
if command -v ipconfig >/dev/null 2>&1; then
    LAN_IP=$(ipconfig getifaddr en0 2>/dev/null)
    if [ -z "$LAN_IP" ]; then
        LAN_IP=$(ipconfig getifaddr en1 2>/dev/null)
    fi
fi
if [ -z "$LAN_IP" ] && command -v hostname >/dev/null 2>&1; then
    LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
if [ -z "$LAN_IP" ] && command -v ifconfig >/dev/null 2>&1; then
    LAN_IP=$(ifconfig 2>/dev/null | awk '/inet / && $2 != "127.0.0.1" {print $2; exit}')
fi

# 3. Run Database Migrations
echo -e "${BLUE}🔄 Running database migrations...${NC}"
# Wait a bit for MySQL to be ready (though healthcheck handles most of it)
sleep 2
docker-compose exec api alembic upgrade head

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️ Migration failed or still waiting for DB. You might need to run 'docker-compose exec api alembic upgrade head' manually later.${NC}"
fi

# 4. Final Status Information
echo -e "\n${GREEN}==============================================${NC}"
echo -e "${GREEN}✨ URITOMO Backend is ready!${NC}"
echo -e "${BLUE}📍 API Base URL: ${NC} http://localhost:8000"
echo -e "${BLUE}📍 API Specs:    ${NC} http://localhost:8000/docs"
echo -e "${BLUE}📊 dashboard Specs:    ${NC} http://localhost:8000/dashboard/"
if [ -n "$LAN_IP" ]; then
    echo -e "${BLUE}🌐 LAN Base URL: ${NC} http://${LAN_IP}:8000"
    echo -e "${BLUE}🌐 LAN Specs:    ${NC} http://${LAN_IP}:8000/docs"
else
    echo -e "${YELLOW}⚠️  LAN IP not detected. Check with: ${NC} ipconfig getifaddr en0"
fi
echo -e "${YELLOW}💡 LAN IP Check: ${NC} ipconfig getifaddr en0"
echo -e "${GREEN}==============================================${NC}"
echo -e "${YELLOW}💡 To see real-time logs, run: ${NC} docker-compose logs -f api"
echo -e "${YELLOW}💡 LiveKit realtime logs:     ${NC} docker-compose logs -f livekit_realtime_agent"
echo -e "${YELLOW}💡 Worker logs (optional):    ${NC} docker-compose --profile with-worker logs -f worker"
echo -e "${YELLOW}💡 To stop services, run:      ${NC} docker-compose down"
