#!/usr/bin/env python3
import sqlite3
import sys
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 odom_bag_to_tum.py <bag_folder> <output_tum.txt>")
        sys.exit(1)

    bag_folder = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    db_files = sorted(bag_folder.glob("*.db3"))
    if not db_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_folder}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fout:
        fout.write("#timestamp x y z q_x q_y q_z q_w\n")

        total_count = 0

        for db_file in db_files:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            cursor.execute("SELECT id, name, type FROM topics")
            topics = cursor.fetchall()

            odom_topic_ids = []
            for topic_id, name, msg_type in topics:
                if name == "/Odometry" and msg_type == "nav_msgs/msg/Odometry":
                    odom_topic_ids.append(topic_id)

            if not odom_topic_ids:
                print(f"No /Odometry topic in {db_file}")
                conn.close()
                continue

            for topic_id in odom_topic_ids:
                cursor.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
                    (topic_id,)
                )

                for _, data in cursor.fetchall():
                    msg = deserialize_message(data, Odometry)

                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation

                    fout.write(
                        f"{t:.9f} "
                        f"{p.x:.15f} {p.y:.15f} {p.z:.15f} "
                        f"{q.x:.15f} {q.y:.15f} {q.z:.15f} {q.w:.15f}\n"
                    )
                    total_count += 1

            conn.close()

    print(f"Saved {total_count} poses to {output_path}")


if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()
