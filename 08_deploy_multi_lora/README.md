# Deploy Multi-LoRA with Llama 3

## Install the Anyscale CLI

```
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```
git clone https://github.com/anyscale/examples.git
cd examples/08_deploy_multi_lora
```

Deploy the service.

```
anyscale service deploy -f service.yaml --env HF_TOKEN=$HF_TOKEN
```

## Query the service

The `anyscale service deploy` command outputs a line that looks like

```
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

From this, you can parse out the service token and base URL. Fill them in in the appropriate location in `query.py`. Then query the model with

```
python query.py
```
