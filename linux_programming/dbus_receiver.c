#include <gio/gio.h>
#include <stdio.h>

// This function is called when the signal is received
static void on_signal_received(GDBusConnection *connection,
                               const gchar *sender_name,
                               const gchar *object_path,
                               const gchar *interface_name,
                               const gchar *signal_name,
                               GVariant *parameters,
                               gpointer user_data) {
    const gchar *message;
    
    // Extract the string from the signal parameters
    g_variant_get(parameters, "(&s)", &message);
    
    printf("Received Signal!\n");
    printf(" - From: %s\n", sender_name);
    printf(" - Payload: %s\n", message);
}

int main() {
    GMainLoop *loop;
    GDBusConnection *conn;
    GError *error = NULL;

    // print pid
    printf("Process PID: %d\n", getpid());

    // 1. Connect to the Session Bus
    conn = g_bus_get_sync(G_BUS_TYPE_SESSION, NULL, &error);
    if (!conn) {
        fprintf(stderr, "Error connecting to D-Bus: %s\n", error->message);
        return 1;
    }

    // 2. Subscribe to the signal
    g_dbus_connection_signal_subscribe(
        conn,
        NULL,                   // Any sender
        "com.example.Demo",     // Interface name
        "HelloWorld",           // Signal name
        "/com/example/Test",    // Object path
        NULL,                   // No arg0 filtering
        G_DBUS_SIGNAL_FLAGS_NONE,
        on_signal_received,
        NULL,                   // User data
        NULL                    // Destroy notify
    );

    printf("Receiver is running. Waiting for signals...\n");

    // 3. Run the main loop
    loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    return 0;
}
