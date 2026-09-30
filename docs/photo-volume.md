# Photo volume

The Pi boots from one SD card, and its root partition fills the card. Without a
separate volume, photos and the operating system share one filesystem, so a
month of 12 MP captures with DNGs (about 14 GB by 2026-09-30) grows towards
filling root. `scripts/pi_photo_volume.py` moves `~/Pictures/pifilm` onto a
preallocated ext4 image file mounted over the same path. Nothing that reads or
writes photos changes: capture, the Stick, `captures.jsonl` and the Nextcloud
sync all keep the same folder.

Why an image file and not a partition: shrinking the root partition needs the
card offline and a Linux machine to resize ext4. An image created with
`fallocate` reserves its space up front, so it gives the same guarantee (photos
can never take more than the image from root) with no repartitioning.

## Run it

On the Pi, from the repo, over Ethernet:

```sh
python3 scripts/pi_photo_volume.py                 # dry run: prints the 22 steps
sudo python3 scripts/pi_photo_volume.py --apply    # does them
```

Defaults: user `george`, folder `/home/george/Pictures/pifilm`, image
`/var/lib/pifilm/photos.img`, 30 GiB (`--user`, `--mount-point`, `--image`,
`--size-gib`). It refuses when the folder is already a mount point, the image or
an fstab entry already exists, or root lacks the image size plus 2 GiB, because
the old copy stays on root until you delete it.

Copying and then checksum-verifying 14 GB on an SD card takes a while. Capture
and the Nextcloud sync are stopped for the whole run. If the verification
finds any difference, it stops before the old folder is touched.

## What it leaves behind

- `/var/lib/pifilm/photos.img`, mode 600, label `pifilm-photos`, no reserved
  blocks.
- An fstab line `… ext4 loop,noatime,nofail,X-fstrim.notrim 0 0`. `nofail`
  means a damaged image cannot stop the Pi booting. `X-fstrim.notrim` keeps the
  weekly `fstrim.timer` off it. Pass `0` skips fsck, which expects a block
  device.

The image must stay fully allocated for its size to be reserved. Trimming a
loop mount punches holes in the backing file and hands that space back to root.
`mkfs` is run with `-E nodiscard` because its default discard does exactly that:
the first run on 2026-09-30, without the flag, left a 30 GiB image with only
15 GiB allocated. Check with `du -h /var/lib/pifilm/photos.img`, which should
show the full size. If it doesn't, `sudo fallocate -l 30G
/var/lib/pifilm/photos.img` fills the holes without touching the data.
- `RequiresMountsFor=` drop-ins for `pifilm-capture.service` and
  `pifilm-nextcloud-sync.service`: neither starts without the volume.
- The bare folder under the mount is immutable (`chattr +i`). If the image
  ever fails to mount, a write into the empty folder fails instead of quietly
  filling root again.
- `~/Pictures/pifilm.pre-volume`, the old copy. Delete it once a capture has
  landed on the new volume: `sudo rm -rf ~/Pictures/pifilm.pre-volume`.

rsync keeps modification times and rclone compares size and time, so the sync
does not re-upload anything after the move.

## Check and grow

```sh
findmnt ~/Pictures/pifilm
df -h ~/Pictures/pifilm /
```

To grow it later, with capture and sync stopped:

```sh
sudo umount ~/Pictures/pifilm
sudo fallocate -l 40G /var/lib/pifilm/photos.img
sudo e2fsck -f /var/lib/pifilm/photos.img && sudo resize2fs /var/lib/pifilm/photos.img
sudo mount ~/Pictures/pifilm
```

## Undo

Stop capture and sync, then:

```sh
sudo umount ~/Pictures/pifilm
sudo chattr -i ~/Pictures/pifilm && rmdir ~/Pictures/pifilm
```

Move `pifilm.pre-volume` back, or copy the image's contents out first if newer
photos are on it. Then remove the fstab line and the two `photo-volume.conf`
drop-ins, and run `sudo systemctl daemon-reload`.
