import sys
from pipeline import Pipeline
from pipeline import parse_arguments

def main():
    try:
        config = parse_arguments()

        if not config.video_path.exists():
            print(f"Error: Video file {config.video_path} does not exist")
            sys.exit(1)

        pipeline = Pipeline(config)
        pipeline.initialize_components()
        pipeline.run()

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()