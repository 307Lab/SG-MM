import os
import argparse
import time
import rospy
from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties, SetPhysicsPropertiesRequest
from std_srvs.srv import Empty
from subprocess import Popen
import rospkg

"""
IMPORTANT: ENSURE THAT THIS FILE ONLY RELIES ON PYTHON2 COMPATIBLE SYNTAX
"""

gazebo_cmds = {
    'husky_ur5': "roslaunch modulation_rl n2m2_husky_empty_world.launch".split(" "),
}


def get_world_file(task, algo):
    rospack = rospkg.RosPack()
    if task in ["rndstartrndgoal", "restrictedws", "simpleobstacle", "spline"]:
        return "empty.world"
    elif task in ["picknplace", "picknplacedyn", "door", "drawer", "roomdoor"]:
        return "modulation_tasks.world"
    elif task in ["dynobstacle"]:
        return "dynamic_world.world"
    elif task in ["bookstorepnp", "bookstoredoor"]:
        world_file = "bookstore_simple.world" if algo in ["moveit", "bi2rrt"] else "bookstore.world"

        if algo not in ["moveit", "bi2rrt"]:
            # the planning_scene plugin will fail with some of the collision meshes
            plugin_path = rospack.get_path("modulation_rl").replace("/src/modulation_rl", "") + "/" + "devel/lib/libgazebo_ros_moveit_planning_scene.so"
            if os.path.exists(plugin_path):
                os.remove(plugin_path)

        # return rospack.get_path("aws-robomaker-bookstore-world-ros1") + "/" + "worlds" + "/" + world_file
        return "/home/ros/kjx/LLB/new_n2m2/src/modulation_rl/gazebo_world/worlds/bookstore.world"
    elif task in ["apartment"]:
        return rospack.get_path("aws-robomaker-bookstore-world-ros1") + "/" + "worlds" + "/" + "small_house.world"
    else:
        raise ValueError("No world defined for task " + task)


def gazebo_set_pyhsics_properties(time_step):
    rospy.init_node('hanlde_launchfiles')
    ns = "gazebo"
    rospy.wait_for_service(ns + '/get_physics_properties')
    rospy.wait_for_service(ns + '/set_physics_properties')

    try:
        get_physics_properties = rospy.ServiceProxy(ns + '/get_physics_properties', GetPhysicsProperties)
        props = get_physics_properties()
    except rospy.ServiceException as e:
        rospy.logerr('couldn\'t get physics properties while preparing to set physics properties: ' + str(e))
        assert False

    try:
        pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        req = SetPhysicsPropertiesRequest(time_step=time_step,
                                          max_update_rate = int(1. / time_step),
                                          gravity=props.gravity,
                                          ode_config=props.ode_config)
        set_physics_properties = rospy.ServiceProxy(ns + '/set_physics_properties', SetPhysicsProperties)

        pause_physics_srv()
        time.sleep(0.01)

        set_physics_properties(req)
        time.sleep(0.01)
        unpause_physics_srv()
    except rospy.ServiceException as e:
        rospy.logerr('couldn\'t set physics properties: ' + str(e))
        assert False
    rospy.signal_shutdown("handle_launchfiles")


def start_launch_files(env_name, algo, task, gui):

    gazebo_cmd = gazebo_cmds[env_name]

    gui = "true" if gui else "false"
    gazebo_cmd += ["gui:=" + gui]

    if (env_name == 'husky_ur5') and (task == "roomdoor"):
        gazebo_cmd += ["local_costmap_frame:=" + "base_link"]

    if (task in ["picknplace", "picknplacedyn", "door", "drawer", "roomdoor"]):
        rospy.set_param('/moving_obstacle/goal_range', 3)

    if (task in ["picknplacedyn", "dynobstacle", "bookstorepnp"]):
        local_map_inflation = 0.03
    else:
        local_map_inflation = 0.0
    rospy.set_param('/costmap_node/costmap/inflation_layer/enabled', local_map_inflation > 0.0)
    rospy.set_param('/costmap_node/costmap/inflation_layer/inflation_radius', local_map_inflation)
    rospy.set_param('/costmap_node/costmap/inflation_radius', local_map_inflation)
    # fast_empty for fast physics, but not sure node will be able to fully keep up
    world_name = get_world_file(task, algo)
    gazebo_cmd += ['world_name:=' + world_name]

    print("Starting command ", gazebo_cmd)
    p_gazebo = Popen(gazebo_cmd) if gazebo_cmd else None
    p_moveit = None

    time.sleep(30)

    # NOTE: THE GRIPPER WILL FAIL TO OPEN / CLOSE WITH THIS STEP SIZE
    # works in a faster world, otherwise it will take forever
    gazebo_set_pyhsics_properties(time_step=0.002)

    return p_gazebo, p_moveit


def stop_launch_files(p_gazebo, p_moveit):
    time.sleep(10)
    if p_gazebo:
        p_gazebo.terminate()
    if p_moveit:
        p_moveit.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str.lower, choices=['pr2', 'tiago', 'hsr', 'husky_ur5'], help='')
    parser.add_argument('--algo', type=str.lower, help='')
    parser.add_argument('--gui', action='store_true', default=False, help='')
    parser.add_argument('--task', default='', type=str.lower, help='')
    args = parser.parse_args()

    assert args.env, "No env supplied for startup. Make sure to start directly through the runfile"

    print("starting roscore")
    p_roscore = Popen(["roscore"])
    time.sleep(5)

    start_launch_files(args.env, args.algo, args.task, args.gui)
    print("\nAll launchfiles started\n")
