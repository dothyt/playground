#include <gio/gio.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    GDBusConnection *conn;
    GVariant *result;
    GError *error = NULL;
    const gchar *echo_str = (argc > 1) ? argv[1] : "Hello D-Bus!";

    conn = g_bus_get_sync(G_BUS_TYPE_SESSION, NULL, &error);

    // Call the Echo method
    result = g_dbus_connection_call_sync(conn,
                                         "com.example.EchoService", // Bus name
                                         "/com/example/EchoObject",  // Object path
                                         "com.example.EchoInterface",// Interface
                                         "Echo",                     // Method
                                         g_variant_new("(s)", echo_str),
                                         G_VARIANT_TYPE("(s)"),      // Expected return
                                         G_DBUS_CALL_FLAGS_NONE,
                                         -1, NULL, &error);

    if (error) {
        fprintf(stderr, "Error calling Echo: %s\n", error->message);
        return 1;
    }

    const gchar *response;
    g_variant_get(result, "(&s)", &response);
    printf("Server replied: %s\n", response);

    // loop forever
    volatile int count = 0;
    while (true) {
	    count++;
    }

    g_variant_unref(result);
    return 0;
}