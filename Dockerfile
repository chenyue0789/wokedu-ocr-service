FROM node:20-bookworm-slim

WORKDIR /app
COPY package*.json ./
RUN npm install --omit=dev

COPY server.js ./
COPY run.sh /opt/application/run.sh
RUN chmod +x /opt/application/run.sh

ENV NODE_ENV=production
ENV PORT=8000
EXPOSE 8000

CMD ["/opt/application/run.sh"]
