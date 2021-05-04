param([String]$name="117M")

mkdir -Force models/$name

$ProgressPreference = 'SilentlyContinue'
foreach ($filename in @("checkpoint","encoder.json","hparams.json",
			"model.ckpt.data-00000-of-00001",
			"model.ckpt.index","model.ckpt.meta","vocab.bpe"))
{
	$fetch = "$name/$filename"
	echo "fetching $fetch"
	Invoke-WebRequest -OutFile "models/$fetch" -Uri "https://storage.googleapis.com/gpt-2/models/$fetch"
}
