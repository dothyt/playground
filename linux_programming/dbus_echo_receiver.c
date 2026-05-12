#include <gio/gio.h>
#include <stdio.h>

// This function handles the "Echo" method call
static void handle_method_call(GDBusConnection *connection,
                               const gchar *sender,
                               const gchar *object_path,
                               const gchar *interface_name,
                               const gchar *method_name,
                               GVariant *parameters,
                               GDBusMethodInvocation *invocation,
                               gpointer user_data) {
    if (g_strcmp0(method_name, "Echo") == 0) {
        const gchar *input;
        g_variant_get(parameters, "(&s)", &input);
        
        printf("Received: %s\n", input);
        
        // Return the string back to the sender
        g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", input));
    }
}

// Metadata describing the interface (XML format)
static const gchar introspection_xml[] =
    "<node>"
    "  <interface name='com.example.EchoInterface'>"
    "    <method name='Echo'>"
    "      <arg type='s' name='input' direction='in'/>"
    "      <arg type='s' name='output' direction='out'/>"
    "    </method>"
    "  </interface>"
    "</node>";

int main() {
    GMainLoop *loop;
    GDBusConnection *conn;
    GDBusNodeInfo *introspection_data;
    GError *error = NULL;

    printf("Process PID: %d\n", getpid());

    conn = g_bus_get_sync(G_BUS_TYPE_SESSION, NULL, &error);
    introspection_data = g_dbus_node_info_new_for_xml(introspection_xml, NULL);

    static const GDBusInterfaceVTable interface_vtable = {
        handle_method_call, NULL, NULL
    };

    // Register the object at a specific path
    g_dbus_connection_register_object(conn,
                                      "/com/example/EchoObject",
                                      introspection_data->interfaces[0],
                                      &interface_vtable,
                                      NULL, NULL, NULL);

    // Request a name on the bus so the sender can find us
    g_bus_own_name_on_connection(conn, "com.example.EchoService", 
                                 G_BUS_NAME_OWNER_FLAGS_NONE, NULL, NULL, NULL, NULL);

    printf("Echo Service is running. Waiting for calls...\n");
    loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    return 0;
}
