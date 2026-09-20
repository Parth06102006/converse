import http from "node:http";
import { app } from "./app.js";
import { setupRealtimeGateway } from "./realtime.js";

const port = Number(process.env.PORT ?? 4000);
const server = http.createServer(app);

setupRealtimeGateway(server);

server.listen(port, () => {
  console.log(`Converse API server listening on http://localhost:${port}`);
});
