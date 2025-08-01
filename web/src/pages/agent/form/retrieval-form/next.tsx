import { Collapse } from '@/components/collapse';
import { CrossLanguageFormField } from '@/components/cross-language-form-field';
import { FormContainer } from '@/components/form-container';
import { KnowledgeBaseFormField } from '@/components/knowledge-base-item';
import { RerankFormFields } from '@/components/rerank';
import { SimilaritySliderFormField } from '@/components/similarity-slider';
import { TopNFormField } from '@/components/top-n-item';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Textarea } from '@/components/ui/textarea';
import { UseKnowledgeGraphFormField } from '@/components/use-knowledge-graph-item';
import { zodResolver } from '@hookform/resolvers/zod';
import { memo, useMemo } from 'react';
import { useForm, useFormContext } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';
import { initialRetrievalValues } from '../../constant';
import { useWatchFormChange } from '../../hooks/use-watch-form-change';
import { INextOperatorForm } from '../../interface';
import { FormWrapper } from '../components/form-wrapper';
import { Output } from '../components/output';
import { QueryVariable } from '../components/query-variable';
import { useValues } from './use-values';

export const RetrievalPartialSchema = {
  similarity_threshold: z.coerce.number(),
  keywords_similarity_weight: z.coerce.number(),
  top_n: z.coerce.number(),
  top_k: z.coerce.number(),
  kb_ids: z.array(z.string()),
  rerank_id: z.string(),
  empty_response: z.string(),
  cross_languages: z.array(z.string()),
  use_kg: z.boolean(),
};

export const FormSchema = z.object({
  query: z.string().optional(),
  ...RetrievalPartialSchema,
});

export function EmptyResponseField() {
  const { t } = useTranslation();
  const form = useFormContext();

  return (
    <FormField
      control={form.control}
      name="empty_response"
      render={({ field }) => (
        <FormItem>
          <FormLabel tooltip={t('chat.emptyResponseTip')}>
            {t('chat.emptyResponse')}
          </FormLabel>
          <FormControl>
            <Textarea
              placeholder={t('common.namePlaceholder')}
              {...field}
              autoComplete="off"
              rows={4}
            />
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

function RetrievalForm({ node }: INextOperatorForm) {
  const outputList = useMemo(() => {
    return [
      {
        title: 'formalized_content',
        type: initialRetrievalValues.outputs.formalized_content.type,
      },
    ];
  }, []);

  const defaultValues = useValues(node);

  const form = useForm({
    defaultValues: defaultValues,
    resolver: zodResolver(FormSchema),
  });

  useWatchFormChange(node?.id, form);

  return (
    <Form {...form}>
      <FormWrapper>
        <FormContainer>
          <QueryVariable></QueryVariable>
          <KnowledgeBaseFormField></KnowledgeBaseFormField>
        </FormContainer>
        <Collapse title={<div>Advanced Settings</div>}>
          <FormContainer>
            <SimilaritySliderFormField
              vectorSimilarityWeightName="keywords_similarity_weight"
              isTooltipShown
            ></SimilaritySliderFormField>
            <TopNFormField></TopNFormField>
            <RerankFormFields></RerankFormFields>
            <EmptyResponseField></EmptyResponseField>
            <CrossLanguageFormField name="cross_languages"></CrossLanguageFormField>
            <UseKnowledgeGraphFormField name="use_kg"></UseKnowledgeGraphFormField>
          </FormContainer>
        </Collapse>
        <Output list={outputList}></Output>
      </FormWrapper>
    </Form>
  );
}

export default memo(RetrievalForm);
