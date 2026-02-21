import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# For OpenAI API api key
api_key = os.getenv("OPENAI_API_KEY")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation tasks.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "backchannel",
            "pause_handling",
            "smooth_turn_taking",
            "user_interruption",
            "behavior",
            "general_before_after",
        ],
        help="Evaluation task to perform.",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing data for evaluation.",
    )

    args = parser.parse_args()

    if args.task == "backchannel":
        from eval_backchannel import eval_backchannel

        eval_backchannel(args.root_dir)
    elif args.task == "pause_handling":
        from eval_pause_handling import eval_pause_handling

        eval_pause_handling(args.root_dir)
    elif args.task == "smooth_turn_taking":
        from eval_smooth_turn_taking import eval_smooth_turn_taking

        eval_smooth_turn_taking(args.root_dir)
    elif args.task == "user_interruption":
        from eval_user_interruption import eval_user_interruption

        client = OpenAI(
            # organization=organization,
            api_key=api_key,
        )
        client.models.list()
        eval_user_interruption(args.root_dir, client)

    elif args.task == "general_before_after":
        from eval_general_before_after import eval_general_all_split

        config = {
            "trim_silence": True,
            "squim": False,
            "utmosv2": True,
            "speaking_rate": True,
            "trim_mode": "silero",
            "agg": {
                "mode": "trim",
                "trim_prop": 0.05,
            },
            "pitch": True,
            "intensity": True,
        }

        aggregate = True
        output = eval_general_all_split(config, args.root_dir, aggregate=aggregate)
        print("General Evaluation Output:", output)
        if aggregate:
            # save to log file
            dir_name = args.root_dir.split("/")[-1]
            log_path = f"{dir_name}_general.log"
            with open(log_path, "w", encoding="utf-8") as f:
                for key, value in output.items():
                    line = f"{key}: {value}\n"
                    print(line, end="")
                    f.write(line)

    elif args.task == "behavior":
        from eval_behavior import eval_behavior_all

        client = OpenAI(
            # organization=organization,
            api_key=api_key,
        )
        output = eval_behavior_all(
            args.root_dir, client, task=args.task, aggregate=True
        )

        dir_name = args.root_dir.split("/")[-1]
        log_path = f"{dir_name}_{args.task}.log"

        with open(log_path, "w", encoding="utf-8") as f:
            for ax in ["C"]:
                line = f"Ratios ({ax}-axis): {output[ax]}\n"
                print(line, end="")
                f.write(line)


if __name__ == "__main__":
    main()
