import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import time
from multiprocessing import Pool
from datetime import datetime




def exec_gemini_call(iteration):
    project_id = "cabral-apigee" 

    vertexai.init(project=project_id, api_endpoint="https://dev.35.227.240.213.nip.io/active-retry", api_transport='rest')
    model = GenerativeModel(model_name="gemini-1.5-pro-001")
    config = GenerationConfig(
        max_output_tokens=100, temperature=0.4, top_p=1, top_k=32
    )
    model.generate_content(
        f"What's a good name for a flower shop that specializes in selling bouquets of {iteration} dried flowers?", generation_config=config
    )

def main():

    iterations = range(100)

    start_time = time.time()
    print(f" Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    with Pool() as pool:
        _ = pool.map(exec_gemini_call, iterations)

    end_time = time.time()
    print(f" End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
