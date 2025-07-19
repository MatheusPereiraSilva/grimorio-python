import os
os.environ["HF_HOME"] = "YOUR_DIRECTORY" #Use this directory to save AI archives.
import discord
from discord.ext import commands
from diffusers import StableDiffusionPipeline
import torch

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@bot.event
async def on_ready():
  print(f'Bot conectado como {bot.user}')

@bot.command(name="gerar")
async def gerar(ctx, *, prompt: str):
  await ctx.send(f"🎨 Grimley esta trabalhando para gerar uma imagem... (ela virá toda errada o Grimely não trabalha)")

  image = pipe(prompt).images[0]
  image.save("output.png")

  await ctx.send(file=discord.File("output.png"))

bot.run("YOUR_TOKEN")
