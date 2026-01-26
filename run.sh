#!/bin/bash

# URITOMO Backend simple runner
# This script works on Mac, Linux, and Git Bash on Windows.

# Text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting URITOMO Backend services...${NC}"

# 1. Start Docker containers in background
# --build: Ensures changes to Dockerfile or app code are reflected
docker-compose up -d --build

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to start docker containers. Make sure Docker Desktop is running.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Containers are up and running.${NC}"

# 2. Determine LAN IP for display (best-effort on macOS)
LAN_IP="10.0.255.246"

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
echo -e "${BLUE}📍 API Base URL: ${NC} http://localhost:8007"
echo -e "${BLUE}📍 API Specs:    ${NC} http://localhost:8007/docs"
if [ -n "$LAN_IP" ]; then
    echo -e "${BLUE}🌐 LAN Base URL: ${NC} http://${LAN_IP}:8007"
    echo -e "${BLUE}🌐 LAN Specs:    ${NC} http://${LAN_IP}:8007/docs"
else
    echo -e "${YELLOW}⚠️  LAN IP not detected. Check with: ${NC} ipconfig getifaddr en0"
fi
echo -e "${YELLOW}💡 LAN IP 확인: ${NC} ipconfig getifaddr en0"
echo -e "${GREEN}==============================================${NC}"
echo -e "${YELLOW}💡 To see real-time logs, run: ${NC} docker-compose logs -f api"
echo -e "${YELLOW}💡 To stop services, run:      ${NC} docker-compose down"
