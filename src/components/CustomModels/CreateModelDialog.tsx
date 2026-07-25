/**
 * "Create Custom Model" dialog: two-step wizard.
 *
 * Step 1: Form — basic model info + engine type selection.
 *         Generates YAML + Python skeleton from form values.
 * Step 2: Code editor — Monaco editor showing YAML (read-only) +
 *         Python code (editable). Save writes files via Tauri command.
 */
import { useState, useMemo } from 'react';
import { Modal, Form, Input, InputNumber, Select, Slider, Radio, Steps, Button, Space, message } from 'antd';
import { useTranslation } from 'react-i18next';
import Editor from '@monaco-editor/react';
import { formatError } from '../../lib/logger';
import { getCustomBridge } from './bridge';

const { TextArea } = Input;

interface FormValues {
  id: string;
  name: string;
  description: string;
  sizeMb: number;
  languages: string[];
  speed: number;
  accuracy: number;
  engineType: 'pip' | 'local';
  pythonModule: string;
  pipPackages: string;
  loadFunction: string;
  transcribeMethod: string;
  languageParam: string;
  pipImportName: string;
  hfRepos: string;
  hasDownload: boolean;
}

function generateYAML(v: FormValues): string {
  const langs: Record<string, string> = {};
  if (v.languages.includes('zh')) langs['zh'] = '中文';
  if (v.languages.includes('en')) langs['en'] = 'English';

  const lines: string[] = [
    `schema_version: 1`,
    `id: ${v.id}`,
    `name: ${v.name}`,
  ];
  if (v.description.trim()) {
    lines.push(`description: "${v.description.trim()}"`);
  }
  lines.push(`size_mb: ${v.sizeMb}`);
  lines.push(`languages:`);
  for (const [code, label] of Object.entries(langs)) {
    lines.push(`  ${code}: ${label}`);
  }
  lines.push(`speed: ${v.speed}`);
  lines.push(`accuracy: ${v.accuracy}`);
  lines.push(`recommended: false`);
  lines.push(``);
  lines.push(`python_module: ${v.pythonModule}`);

  if (v.engineType === 'pip' && v.pipPackages.trim()) {
    lines.push(`pip_packages:`);
    for (const pkg of v.pipPackages.split('\n').map(s => s.trim()).filter(Boolean)) {
      lines.push(`  - ${pkg}`);
    }
    if (v.pipImportName.trim()) {
      lines.push(`pip_import_name: ${v.pipImportName.trim()}`);
    }
  }

  if (v.hasDownload && v.hfRepos.trim()) {
    lines.push(``);
    lines.push(`download:`);
    lines.push(`  hf_repos:`);
    for (const repo of v.hfRepos.split('\n').map(s => s.trim()).filter(Boolean)) {
      lines.push(`    - ${repo}`);
    }
    lines.push(`  paths:`);
    lines.push(`    model_dir: "{repo_dirs[0]}"`);
  }

  lines.push(``);
  lines.push(`load:`);
  lines.push(`  function: ${v.loadFunction}`);
  lines.push(`  kwargs: {}`);
  lines.push(``);
  lines.push(`transcribe_method: ${v.transcribeMethod}`);
  lines.push(`language_param: ${v.languageParam}`);

  return lines.join('\n') + '\n';
}

function generatePluginCode(v: FormValues): string {
  const mod = v.pythonModule;
  return `"""${v.name} — Custom ASR plugin for VoiceInk (语墨).

This plugin loads the model directly in the daemon process.
Implement load() to return a model object with a transcribe/generate method.
"""

import sys
from pathlib import Path


def load(model_dir="", vad_model_dir=""):
    """Load the model and return an object with a .${v.transcribeMethod}() method.

    Args:
        model_dir: Path to model weights (from download step, if any).
        vad_model_dir: Optional VAD model path. Empty = no VAD.

    Returns:
        A model object. The daemon will call model.${v.transcribeMethod}(audio_path, language=...)
        to transcribe each audio file.
    """
    # TODO: import your model library and create an instance
    # Example:
    # from your_package import YourModel
    # return YourModel(model_path=model_dir)

    raise NotImplementedError("Implement load() in ${mod}/__init__.py")
`;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
  /** If set, open in edit mode — skip form, load existing YAML + code. */
  editSpecPath?: string | null;
}

export function CreateModelDialog({ open, onClose, onCreated, editSpecPath }: Props) {
  const { t } = useTranslation();
  const isEdit = !!editSpecPath;
  const [step, setStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [form] = Form.useForm<FormValues>();
  const [yamlContent, setYamlContent] = useState('');
  const [pluginCode, setPluginCode] = useState('');
  const [engineType, setEngineType] = useState<'pip' | 'local'>('local');

  // In edit mode, load existing content and jump to step 1 (code editor)
  useMemo(() => {
    if (open && isEdit && editSpecPath) {
      getCustomBridge().invoke('custom-read-model', editSpecPath).then((res) => {
        const result = res as { yaml: string; pluginCode: string | null };
        setYamlContent(result.yaml);
        setPluginCode(result.pluginCode || '');
        setStep(1);
      }).catch((e) => {
        message.error(formatError(e, t('customModels.readFailed')));
      });
    }
  }, [open, isEdit, editSpecPath, t]);
  const handleNext = () => {
    form.validateFields().then((values) => {
      const merged = { ...values, engineType } as FormValues;
      setYamlContent(generateYAML(merged));
      if (merged.engineType === 'local') {
        setPluginCode(generatePluginCode(merged));
      } else {
        setPluginCode('');
      }
      setStep(1);
    });
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      if (isEdit && editSpecPath) {
        await getCustomBridge().invoke(
          'custom-save-model',
          editSpecPath,
          yamlContent,
          pluginCode || undefined,
        );
        message.success(t('customModels.saveSuccess'));
      } else {
        await getCustomBridge().invoke(
          'custom-create-model',
          yamlContent,
          pluginCode || undefined,
        );
        message.success(t('customModels.createSuccess'));
      }
      onCreated();
      handleClose();
    } catch (e) {
      message.error(formatError(e, t('customModels.createFailed')));
    } finally {
      setSaving(false);
    }
  };

  const handleClose = () => {
    setStep(0);
    form.resetFields();
    setYamlContent('');
    setPluginCode('');
    onClose();
  };

  return (
    <Modal
      open={open}
      onCancel={handleClose}
      width={900}
      footer={null}
      title={t('customModels.createModel')}
      destroyOnClose
    >
      <Steps
        current={step}
        size="small"
        style={{ marginBottom: 24 }}
        items={[
          { title: t('customModels.createStep1') },
          { title: t('customModels.createStep2') },
        ]}
      />

      {step === 0 && (
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            sizeMb: 0,
            languages: ['zh', 'en'],
            speed: 7,
            accuracy: 8,
            transcribeMethod: 'transcribe',
            languageParam: 'language',
            loadFunction: '',
            hasDownload: true,
          }}
        >
          <Form.Item name="id" label={t('customModels.field.id')} rules={[{ required: true }]}>
            <Input placeholder="custom-my-asr" />
          </Form.Item>
          <Form.Item name="name" label={t('customModels.field.name')} rules={[{ required: true }]}>
            <Input placeholder="My ASR" />
          </Form.Item>
          <Form.Item name="description" label={t('customModels.field.description')}>
            <TextArea rows={2} />
          </Form.Item>

          <Space style={{ width: '100%' }} size="middle">
            <Form.Item name="sizeMb" label={t('customModels.field.size')}>
              <InputNumber min={0} addonAfter="MB" style={{ width: 180 }} />
            </Form.Item>
            <Form.Item name="languages" label={t('customModels.field.languages')}>
              <Select
                mode="multiple"
                style={{ width: 240 }}
                options={[
                  { value: 'zh', label: '中文' },
                  { value: 'en', label: 'English' },
                ]}
              />
            </Form.Item>
          </Space>

          <Space style={{ width: '100%' }} size="middle">
            <Form.Item name="speed" label={t('customModels.field.speed')}>
              <Slider min={1} max={10} style={{ width: 200 }} />
            </Form.Item>
            <Form.Item name="accuracy" label={t('customModels.field.accuracy')}>
              <Slider min={1} max={10} style={{ width: 200 }} />
            </Form.Item>
          </Space>

          <Form.Item label={t('customModels.field.engineType')}>
            <Radio.Group value={engineType} onChange={(e) => setEngineType(e.target.value)}>
              <Radio.Button value="local">{t('customModels.engine.local')}</Radio.Button>
              <Radio.Button value="pip">{t('customModels.engine.pip')}</Radio.Button>
            </Radio.Group>
          </Form.Item>

          <Form.Item name="pythonModule" label={t('customModels.field.pythonModule')} rules={[{ required: true }]}>
            <Input placeholder={engineType === 'local' ? 'my_asr (会生成同名文件夹)' : 'mimo_mlx'} />
          </Form.Item>

          {engineType === 'pip' && (
            <>
              <Form.Item name="pipPackages" label={t('customModels.field.pipPackages')}>
                <TextArea rows={2} placeholder={'mimo_mlx>=0.1.0\n或 git+https://github.com/...'} />
              </Form.Item>
              <Form.Item name="pipImportName" label={t('customModels.field.pipImportName')}>
                <Input placeholder="留空则自动推断" />
              </Form.Item>
            </>
          )}

          <Form.Item name="loadFunction" label={t('customModels.field.loadFunction')} rules={[{ required: true }]}>
            <Input placeholder={engineType === 'local' ? 'my_asr.load' : 'mimo_mlx.load_asr'} />
          </Form.Item>

          <Space style={{ width: '100%' }} size="middle">
            <Form.Item name="transcribeMethod" label={t('customModels.field.transcribeMethod')}>
              <Input placeholder="transcribe 或 generate" />
            </Form.Item>
            <Form.Item name="languageParam" label={t('customModels.field.languageParam')}>
              <Input placeholder="language" />
            </Form.Item>
          </Space>

          <Form.Item name="hasDownload" label={t('customModels.field.hasDownload')} valuePropName="checked">
            <Radio.Group>
              <Radio value={true}>{t('customModels.download.hf')}</Radio>
              <Radio value={false}>{t('customModels.download.none')}</Radio>
            </Radio.Group>
          </Form.Item>

          <Form.Item shouldUpdate noStyle>
            {({ getFieldValue }) =>
              getFieldValue('hasDownload') ? (
                <Form.Item name="hfRepos" label={t('customModels.field.hfRepos')}>
                  <TextArea rows={2} placeholder="Mininglamp-2718/Mano-ASR-1.7B-Instruct-1.0-MLX-8bit" />
                </Form.Item>
              ) : null
            }
          </Form.Item>

          <div style={{ textAlign: 'right' }}>
            <Button type="primary" onClick={handleNext}>
              {t('customModels.next')}
            </Button>
          </div>
        </Form>
      )}

      {step === 1 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>YAML</div>
            <Editor
              height="250px"
              language="yaml"
              value={yamlContent}
              onChange={(v) => setYamlContent(v || '')}
              options={{ minimap: { enabled: false }, fontSize: 13, wordWrap: 'on' }}
            />
          </div>
          {pluginCode && (
            <div>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Python ({form.getFieldValue('pythonModule')}/__init__.py)</div>
              <Editor
                height="300px"
                language="python"
                value={pluginCode}
                onChange={(v) => setPluginCode(v || '')}
                options={{ minimap: { enabled: false }, fontSize: 13, wordWrap: 'on' }}
              />
            </div>
          )}
          <div style={{ textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setStep(0)}>{t('customModels.back')}</Button>
              <Button type="primary" loading={saving} onClick={handleSave}>
                {t('customModels.save')}
              </Button>
            </Space>
          </div>
        </div>
      )}
    </Modal>
  );
}

export default CreateModelDialog;
