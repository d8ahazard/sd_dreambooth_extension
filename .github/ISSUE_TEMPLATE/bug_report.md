---

name: Bug report
about: Please Answer the Following Questions
title: ''
labels: 'new'
assignees: ''

---
## ⚠️If you do not follow the template, your issue may be closed without a response ⚠️
Kindly read and fill this form in its entirety.

### 0. Initial troubleshooting
Please check each of these before opening an issue. If you've checked them, **delete this section of your bug report**. Have you:
- Updated the Stable-Diffusion-WebUI to the latest version?
- Updated Dreambooth to the latest revision?
- Completely restarted the stable-diffusion-webUI, not just reloaded the UI?
- Read the [Readme](https://github.com/d8ahazard/sd_dreambooth_extension#readme)?

### 1. Please find the following lines in the console and paste them below.

```
#######################################################################################################
Initializing Dreambooth
If submitting an issue on github, please provide the below text for debugging purposes:

Python revision: 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)]
Dreambooth revision: bd3fecc3d27d777a4e8f3206a0b16e852877dbad
SD-WebUI revision: 

[+] torch version 2.0.0+cu118 installed.
[+] torchvision version 0.15.1+cu118 installed.
[+] xformers version 0.0.17+b6be33a.d20230315 installed.
[+] accelerate version 0.17.1 installed.
[+] bitsandbytes version 0.35.4 installed.
[+] diffusers version 0.14.0 installed.
[+] transformers version 4.27.1 installed.

#######################################################################################################
```


### 2. Describe the bug

(A clear and concise description of what the bug is)

**Screenshots/Config**
If the issue is specific to an error while training, please provide a screenshot of training parameters or the
db_config.json file from /models/dreambooth/MODELNAME/db_config.json

### 3. Provide logs

If a crash has occurred, please provide the *entire* stack trace from the log, including the last few log messages *before* the crash occurred.

```st
PASTE YOUR STACKTRACE HERE
```

### 4. Environment

What OS?

If Windows - WSL or native?

What GPU are you using?
