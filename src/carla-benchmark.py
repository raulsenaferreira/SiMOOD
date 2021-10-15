import carla
import time

# connect to Carla
client = carla.Client("localhost", 2000)
world = client.load_world("Town04")

sync = False

settings = world.get_settings()
settings.fixed_delta_seconds = 0.05
if sync:
    settings.synchronous_mode = True
else:
    settings.synchronous_mode = False
world.apply_settings(settings)

# make first tick
if sync:
    world.tick()
else:
    world.wait_for_tick()

# evaluate tick performance per second
if sync:
    t0 = time.perf_counter()
    ticks0 = world.get_snapshot().frame
    while time.perf_counter() - t0 < 1.0:
        world.tick()
    ticks1 = world.get_snapshot().frame
    print("Made", ticks1-ticks0, "ticks per second!")
else:
    ticks0 = world.get_snapshot().frame
    time.sleep(1)
    ticks1 = world.get_snapshot().frame
    print("Made", ticks1-ticks0, "ticks per second!")