import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import time
from multiprocessing import Pool
from datetime import datetime
import random
from wonderwords import RandomWord



def exec_gemini_call(iteration):
    project_id = "cabral-apigee" 
    API_KEY=["imY9LzbR68tQGk7GED52iNhEne3UYskQf7EAEkAfD6KkrVF6", "lWSakkcBPKRejFb0ap7ebxuADWsDAdJrFdjTwgPC7v2ch75Z"]

    key=random.choices(API_KEY, weights=[0.3, 0.7], k=1)[0]
    vertexai.init(project=project_id, api_endpoint="https://dev.35.227.240.213.nip.io/auth-active-retry", api_transport='rest',  request_metadata=[("x-apikey", key)])
   
    model = GenerativeModel(model_name="gemini-1.5-pro-001")
    config = GenerationConfig(
        max_output_tokens=100, temperature=0.4, top_p=1, top_k=32
    )
    model.generate_content(
        f"Generate a poem that contains the following word: {RandomWord().word(word_min_length=1, word_max_length=25)}", generation_config=config
    )

def main():

    iterations = range(30)

    start_time = time.time()
    print(f" Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    with Pool() as pool:
        _ = pool.map(exec_gemini_call, iterations)

    end_time = time.time()
    print(f" End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
