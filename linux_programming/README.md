### dbus compile command

```
gcc dbus_receiver.c $(pkg-config --cflags --libs gio-2.0) -o receiver
gcc dbus_sender.c $(pkg-config --cflags --libs gio-2.0) -o sender
```