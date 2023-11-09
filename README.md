# tencent-ailab / IP-Adapter

This is an implementation of the [tencent-ailab / IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@ai_face2.png -i prompt="photo of a beautiful girl wearing casual shirt in a garden" -i seed=42

## Example:

"photo of a beautiful girl wearing casual shirt in a garden"

![alt text](output.0.png)
