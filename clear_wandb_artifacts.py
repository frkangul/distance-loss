import wandb

WANDB_KEY="615a4a8c6b3ade78e75eba4a9c1ed70e4f564178"
wandb.login(key=WANDB_KEY)

# Delete unnecessary model files from WANDB artifacts
api = wandb.Api(overrides={
        "project": "foreground-car-segm"
        })

artifact_type, artifact_name = "model", "model-lmwcg972" 
for version in api.artifact_versions(artifact_type, artifact_name):
    print(version.aliases)
    if len(version.aliases) == 0:
        version.delete()