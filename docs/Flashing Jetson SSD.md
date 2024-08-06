# Flashing Jetson SSD

**Given:**

- Auvidia JNX30D board needs custom drivers
- Auvidia website doesn’t have the latest JetPack image

⇒ download the latest jetpack image with Nvidia SDK manager and patch it with custom auvidia firmware (as described in the “advanced flashing guide” section of the [guide](https://auvidea.eu/download/Software)).

But the eMMC storage is very limited and doesn’t fit even full jetpack itself. 

- One option - remove unnecessary things - [guide](https://dev.to/ajeetraina/how-i-cleaned-up-jetson-nano-disk-space-by-40-b9h);
- another - boot from SSD. Easy to do through the SDK manager (no auvidia drivers though), a bit harder with the [bootFromExternalStorage](https://github.com/jetsonhacks/bootFromExternalStorage) util.

One of the unsolved problems - the default partition table of the SSD would imitate eMMC, limiting the capacity to 16gb. Solution - [here](https://github.com/jetsonhacks/bootFromExternalStorage/issues/40): boot from eMMC and reshuffle the partitions manually (which I did).  Possibly there is a partition table file in the repo/image which dictates how to set up the disk during flashing, which can be edited to suit our needs, but I didn’t look for it too much.