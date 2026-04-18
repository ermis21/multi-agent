FROM node:20-slim
RUN npm install -g @notionhq/notion-mcp-server
ENV NOTION_TOKEN=""
EXPOSE 3000
CMD ["notion-mcp-server", "--transport", "http", "--port", "3000"]
