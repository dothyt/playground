#include <gio/gio.h>
#include <stdio.h>

int main() {
    GDBusConnection *conn;
    GError *error = NULL;

    // 1. Connect to the Session Bus
    conn = g_bus_get_sync(G_BUS_TYPE_SESSION, NULL, &error);
    if (!conn) {
        fprintf(stderr, "Error: %s\n", error->message);
        return 1;
    }

    printf("Sending 'HelloWorld' signal...\n");

    // 2. Emit the signal
    gboolean success = g_dbus_connection_emit_signal(
        conn,
        NULL,                   // Destination (NULL for broadcast)
        "/com/example/Test",    // Object path
        "com.example.Demo",     // Interface
        "HelloWorld",           // Signal name
        g_variant_new("(s)", "Hello from the C Sender!"), // Payload
        &error
    );

    if (!success) {
        fprintf(stderr, "Failed to emit signal: %s\n", error->message);
        return 1;
    }

    // Flush the connection to ensure the message is sent before exiting
    g_dbus_connection_flush_sync(conn, NULL, NULL);
    g_object_unref(conn);

    printf("Signal sent successfully.\n");
    return 0;
}